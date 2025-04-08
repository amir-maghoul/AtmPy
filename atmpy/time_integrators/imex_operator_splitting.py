"""This module contains different time integrators"""

from __future__ import annotations
from typing import TYPE_CHECKING, Union, List, Any

from atmpy.boundary_conditions.bc_extra_operations import WallAdjustment
from atmpy.boundary_conditions.contexts import BCApplicationContext
from atmpy.infrastructure.utility import directional_indices, one_element_inner_slice
from atmpy.pressure_solver.discrete_operations import AbstractDiscreteOperator
from atmpy.infrastructure.enums import VariableIndices as VI

if TYPE_CHECKING:
    from atmpy.flux.flux import Flux
    from atmpy.variables.variables import Variables
    from atmpy.variables.multiple_pressure_variables import MPV
    from atmpy.grid.kgrid import Grid
    from atmpy.boundary_conditions.boundary_manager import BoundaryManager
    from atmpy.physics.gravity import Gravity
    from atmpy.physics.thermodynamics import Thermodynamics
    from atmpy.time_integrators.coriolis import CoriolisOperator
    from atmpy.pressure_solver.pressure_solvers import AbstractPressureSolver

from atmpy.time_integrators.abstract_time_integrator import AbstractTimeIntegrator
from atmpy.time_integrators.utility import *

from atmpy.infrastructure.enums import (
    BoundarySide as BdrySide,
    BoundaryConditions as BdryType,
)


class IMEXTimeIntegrator(AbstractTimeIntegrator):
    def __init__(
        self,
        grid: "Grid",
        variables: "Variables",
        mpv: "MPV",
        flux: "Flux",
        boundary_manager: "BoundaryManager",
        coriolis_operator: "CoriolisOperator",
        pressure_solver: "AbstractPressureSolver",
        thermodynamics: "Thermodynamics",
        dt: float,
        Msq: float,
        is_nongeostrophic: bool,
        is_nonhydrostatic: bool,
        is_compressible: bool,
        wind_speed: float = [0.0, 0.0, 0.0],
    ):
        # Inject dependencies
        self.grid: "Grid" = grid
        self.variables: "Variables" = variables
        self.mpv: "MPV" = mpv
        self.flux: "Flux" = flux
        self.boundary_manager: "BoundaryManager" = boundary_manager
        self.coriolis: "CoriolisOperator" = coriolis_operator
        self.gravity: "Gravity" = self.coriolis.gravity
        self.pressure_solver: "AbstractPressureSolver" = pressure_solver
        self.discrete_operator: "AbstractDiscreteOperator" = (
            self.pressure_solver.discrete_operator
        )
        self.th: "Thermodynamics" = thermodynamics
        self.dt: float = dt
        self.Msq: float = Msq
        self.is_nongeostrophic: bool = is_nongeostrophic
        self.is_nonhydrostatic: bool = is_nonhydrostatic
        self.is_compressible: bool = is_compressible
        self.wind_speed: Union[list, np.ndarray] = wind_speed

    def step(self):
        # 1. Explicit forward update (e.g. divergence, pressure gradient, momentum update)
        self.forward_update()

        # 2. Apply boundary conditions (if needed)
        self.boundary_manager.apply_all(self.variables)

        # 3. Implicit backward update: solve for the pressure correction
        self.backward_update_implicit()

        # 4. Explicit correction update (e.g. adjusting momentum with background winds, applying inverse Coriolis)
        self.backward_update_explicit()

        # 5. Final boundary condition update, if needed.
        self.boundary_manager.apply_all(self.variables)

    def forward_update(self):
        """Integrates the problem on time step using the explicit euler in the NON-ADVECTIVE stage. This means
        that the algorithm does not use the half-time advective values to compute the updates. Rather, we compute
        directly from the full-variables in the main euler equation."""
        g = self.gravity.strength  # The gravity strength
        cellvars = self.variables.cell_vars  # Temp variable for cell variables
        p2n = np.copy(self.mpv.p2_nodes)  # Temp allocation for p2_nodes

        # Calculate the buoyancy PX'
        dbuoy = cellvars[..., VI.RHOY] * cellvars[..., VI.RHOX] / cellvars[..., VI.RHO]

        # index 0: momentum in the direction of gravity. index 1: momentum in the direction of non-gravity.
        vertical_momentum_index = self.gravity.gravity_momentum_index

        ###################### Update variables
        self._forward_momenta_update(cellvars, p2n, dbuoy, g, vertical_momentum_index)
        self._forward_buoyancy_update(cellvars, vertical_momentum_index)
        self._forward_pressure_update(cellvars, p2n)

        ####################### Update boundary values

        # First create the application context for the boundary manager. That is setting the flag is_nodal for all
        # dimensions and all sides (therefore ndim*2) to True since p2_nodes is nodal.
        contexts = [BCApplicationContext(is_nodal=True)] * self.grid.ndim * 2

        # Update the boundary nodes for pressure variable
        self.boundary_manager.apply_boundary_on_single_var_all_sides(
            self.mpv.p2_nodes, contexts
        )

        # Update all other variables.
        self.boundary_manager.apply_boundary_on_all_sides(cellvars)

    def _forward_momenta_update(
        self,
        cellvars: np.ndarray,
        p2n: np.ndarray,
        dbuoy: np.ndarray,
        g: float,
        vertical_momentum_index: int,
    ):
        """Update the momenta

        Parameters
        ----------
        cellvars: np.ndarray
            The cell variables
        p2n: np.ndarray
            The nodal pressure variables
        dbuoy : np.ndarray
            The pressured perturbation of Chi: PX'
        g: float
            The gravity strength
        vertical_momentum_index: VariablesIndices (Enum)
            The index of the vertical momentum in the variable container
        """
        coriolis = self.coriolis.strength
        adjusted_momenta = self.variables.adjust_background_wind(self.wind_speed, -1.0)

        # pressure gradient factor: (P/Gamma)
        rhoYovG = self.variables.cell_vars[..., VI.RHOY] * self.th.Gammainv

        # Calculate the pressure gradiant (RHS of the momenta equations)
        dpdx, dpdy, dpdz = self.discrete_operator.gradient(p2n)

        ###############################################################################################################
        ## UPDATING VARIABLES
        ###############################################################################################################
        # Updates: First the shared terms without regarding which one is in the gravity direction
        # Horizontal momentum in x
        cellvars[..., VI.RHOU] -= self.dt * (
            rhoYovG * dpdx
            - coriolis[2] * adjusted_momenta[..., 1]
            + coriolis[1] * adjusted_momenta[..., 2]
        )
        if self.grid.ndim >= 2:
            cellvars[..., VI.RHOV] -= self.dt * (
                rhoYovG * dpdy
                - coriolis[0] * adjusted_momenta[..., 2]
                + coriolis[2] * adjusted_momenta[..., 0]
            )
        if self.grid.ndim == 3:
            cellvars[..., VI.RHOU] -= self.dt * (
                rhoYovG * dpdy
                - coriolis[1] * adjusted_momenta[..., 0]
                + coriolis[0] * adjusted_momenta[..., 1]
            )

        # Updates: The momentum in the direction of gravity
        # Find vertical vs horizontal velocities:
        cellvars[..., vertical_momentum_index] -= self.dt * (
            (g / self.Msq) * dbuoy * self.is_nongeostrophic
        )

    def _forward_pressure_update(self, cellvars: np.ndarray, p2n: np.ndarray):
        """Update the Exner pressure."""

        # Compute the weighting factor Y = (rhoY / rho) = Theta
        Y = cellvars[..., VI.RHOY] / cellvars[..., VI.RHO]

        # Compute the divergence of the pressure-weighted momenta: (Pu)_x + (Pv)_y + (Pw)_z where
        # P = rho*Y = rho*Theta
        momenta_indices = [VI.RHOU, VI.RHOV, VI.RHOW][: self.grid.ndim]
        inner_slice = one_element_inner_slice(self.grid.ndim, full=False)
        self.mpv.rhs[inner_slice] = self.discrete_operator.divergence(
            self.variables.cell_vars[..., momenta_indices] * Y[..., np.newaxis],
        )

        # Adjust wall boundary nodes (scale). Notice the side is set to be BdrySide.ALL.
        # This will apply the 'extra' method whenever the boundary is defined to be WALL.
        boundary_operation = [
            WallAdjustment(
                target_side=BdrySide.ALL, target_type=BdryType.WALL, factor=2.0
            )
        ]
        self.boundary_manager.apply_extra_all_sides(self.mpv.rhs, boundary_operation)

        # Calculate the derivative of the Exner pressure with respect to P
        dpidP = calculate_dpi_dp(cellvars[..., VI.RHOY], self.Msq)

        # Create node-to-cell index (slice(1, -1) in all directions)
        inner_idx = one_element_inner_slice(self.grid.ndim, full=False)

        # Create a nodal variable to store the intermediate updates
        dp2n = np.zeros_like(p2n)
        dp2n[inner_idx] -= self.dt * dpidP * self.mpv.rhs[inner_slice]
        self.mpv.p2_nodes[...] += self.is_compressible * dp2n

    def _forward_buoyancy_update(
        self, cellvars: np.ndarray, vertical_momentum_index: int
    ):
        """Update the X' variable (rho X in the variables)

        Parameters
        ----------
        cellvars: np.ndarray
           The cell variables
        vertical_momentum_index: VariablesIndices (Enum)
           The index of the vertical momentum in the variable container
        """
        # get Chi variable and the derivative
        S0c = self.mpv.get_S0c_on_cells()
        dSdy = self.mpv.compute_dS_on_nodes()

        # Intermediate variable for current Chi
        currentX = cellvars[..., VI.RHO] * (
            (cellvars[..., VI.RHO] / cellvars[..., VI.RHOY]) - S0c
        )

        ###############################################################################################################
        # Update the variable
        ###############################################################################################################
        cellvars[..., VI.RHOX] = (
            currentX
            - self.dt
            * cellvars[..., vertical_momentum_index]
            * dSdy
            * cellvars[..., VI.RHO]
        )

    def backward_update_implicit(self):
        # Encapsulate the pressure solve (euler_backward_non_advective_impl_part)
        # For example, assemble the pressure operator and call the pressure_solver.
        self.pressure_solver.solve(self.variables, self.dt)
        print("Implicit (pressure) update complete.")

    def backward_update_explicit(self):
        # Encapsulate the explicit corrections like background wind modifications
        # and the explicit multiplicative inverse of the Coriolis operator
        # (euler_backward_non_advective_expl_part)
        self.adjust_background_wind(-1.0)
        self.coriolis_operator.apply(self.variables, self.dt)
        self.adjust_background_wind(+1.0)
        print("Explicit backward update complete.")

    def get_dt(self):
        return self.dt


def example_usage():
    from atmpy.physics.thermodynamics import Thermodynamics
    from atmpy.grid.utility import DimensionSpec, create_grid
    from atmpy.variables.variables import Variables
    from atmpy.infrastructure.enums import VariableIndices as VI
    from atmpy.boundary_conditions.bc_extra_operations import (
        WallAdjustment,
        PeriodicAdjustment,
    )

    np.set_printoptions(linewidth=300)
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=5)

    ####################################################################################################################
    # GRID DATA ########################################################################################################
    ####################################################################################################################

    nx = 6
    ngx = 2
    nnx = nx + 2 * ngx
    ny = 5
    ngy = 2
    nny = ny + 2 * ngy

    dim = [DimensionSpec(nx, 0, 2, ngx), DimensionSpec(ny, 0, 2, ngy)]
    grid = create_grid(dim)

    ####################################################################################################################
    ## VARIABLE DATA ###################################################################################################
    ####################################################################################################################

    rng = np.random.default_rng()
    arr = np.arange(nnx * nny)
    rng.shuffle(arr)
    array = arr.reshape(nnx, nny)

    variables = Variables(grid, 6, 1)
    variables.cell_vars[..., VI.RHO] = 1
    variables.cell_vars[..., VI.RHO][1:-1, 1:-1] = 4
    variables.cell_vars[..., VI.RHOU] = array
    variables.cell_vars[..., VI.RHOY] = 2
    variables.cell_vars[..., VI.RHOW] = 3

    rng.shuffle(arr)
    array = arr.reshape(nnx, nny)
    variables.cell_vars[..., VI.RHOV] = array
    gravity = np.array([0.0, 1.0, 0.0])
    th = Thermodynamics()

    ####################################################################################################################
    ######### MPV ######################################################################################################
    ####################################################################################################################
    from atmpy.variables.multiple_pressure_variables import MPV

    mpv = MPV(grid)

    ####################################################################################################################
    ######### FLUX #####################################################################################################
    ####################################################################################################################
    from atmpy.physics.eos import ExnerBasedEOS
    from atmpy.flux.flux import Flux

    eos = ExnerBasedEOS()
    flux = Flux(grid, variables, eos)

    ####################################################################################################################
    ######### BOUNDARY MANAGER #########################################################################################
    ####################################################################################################################
    from atmpy.boundary_conditions.boundary_manager import BoundaryManager
    from atmpy.boundary_conditions.contexts import (
        BCInstantiationOptions,
        BoundaryConditionsConfiguration,
        BCApplicationContext,
    )

    gravity_vec = [0.0, 1.0, 0.0]
    direction = "y"

    bc = BCInstantiationOptions(
        side=BdrySide.BOTTOM, type=BdryType.WALL, direction=direction, grid=grid
    )
    bc2 = BCInstantiationOptions(
        side=BdrySide.TOP, type=BdryType.WALL, direction=direction, grid=grid
    )
    # bc3 = RFBCInstantiationOptions(
    #     side=BdrySide.LEFT, type=BdryType.WALL, direction="x", grid=grid
    # )
    # bc4 = RFBCInstantiationOptions(
    #     side=BdrySide.RIGHT, type=BdryType.PERIODIC, direction="x", grid=grid
    # )
    # options = [bc, bc2, bc3, bc4]
    options = [bc, bc2]
    bc_conditions = BoundaryConditionsConfiguration(options)
    manager = BoundaryManager(bc_conditions)

    ####################################################################################################################
    ########## DISCRETE OPERATOR AND PRESSURE SOLVER ###################################################################
    ####################################################################################################################
    from atmpy.infrastructure.enums import (
        PressureSolvers,
        DiscreteOperators,
        LinearSolvers,
    )
    from atmpy.pressure_solver.contexts import (
        DCInstantiationContext,
        PSInstantiationContext,
    )

    op_context = DCInstantiationContext(ndim=grid.ndim, dxyz=grid.dxyz)
    linear_solver = LinearSolvers.BICGSTAB

    # Instantiate the pressure solver context by specifying enums for pressure solver and discrete operator.
    ps_context = PSInstantiationContext(
        solver_type=PressureSolvers.CLASSIC_PRESSURE_SOLVER,
        discrete_operator_type=DiscreteOperators.CLASSIC_OPERATOR,
        op_context=op_context,
        linear_solver_type=linear_solver,
    )

    pressure = ps_context.instantiate()

    ####################################################################################################################
    ######## TIME INTEGRATION ##########################################################################################
    ####################################################################################################################
    from atmpy.physics.gravity import Gravity
    from atmpy.time_integrators.coriolis import CoriolisOperator

    gravity = Gravity(gravity_vec, grid.ndim)
    coriolis = CoriolisOperator([0.0, 1.0, 0.0], gravity)

    ##### Approach 1 ######################################################
    ### Using instantiation context for central instatiation of all classes:
    from atmpy.time_integrators.contexts import TimeIntegratorContext
    from atmpy.infrastructure.enums import TimeIntegrators

    context = TimeIntegratorContext(
        integrator_type=TimeIntegrators.IMEX,
        grid=grid,
        variables=variables,
        flux=flux,
        boundary_manager=manager,
        dt=0.1,
        extra_dependencies={
            "mpv": mpv,
            "coriolis_operator": coriolis,
            "pressure_solver": pressure,
            "thermodynamics": th,
            "is_nonhydrostatic": True,
            "is_nongeostrophic": True,
            "is_compressible": True,
            "Msq": 1.0,
            "wind_speed": [0.0, 0.0, 0.0],  # optional: override default wind speed
        },
    )
    time_integrator = context.instantiate()

    ###### Approach 2 #############################
    # Or simply using the direct needed integrator:
    # time_integrator = IMEXTimeIntegrator(
    #     grid=grid,
    #     variables=variables,
    #     mpv=mpv,
    #     flux=flux,
    #     boundary_manager=manager,
    #     coriolis_operator=coriolis,
    #     pressure_solver=pressure,
    #     thermodynamics=th,
    #     dt=0.1,
    #     Msq=1.0
    # )
    print(variables.cell_vars[..., VI.RHOU])
    time_integrator.forward_update()
    time_integrator.forward_update()

    print(variables.cell_vars[..., VI.RHOU])


if __name__ == "__main__":
    example_usage()
