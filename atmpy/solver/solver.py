# solver.py

from typing import Any
from atmpy.configuration.simulation_configuration import SimulationConfig
from atmpy.configuration.simulation_data import SpatialGrid, Temporal
from atmpy.boundary_conditions.boundary_manager import BoundaryConditionManager
from atmpy.flux.flux import Flux
from atmpy.grid.kgrid import Grid
from atmpy.variables.variables import Variables
from atmpy.test_cases.traveling_vortex import TravelingVortex
import numpy as np


class Solver:
    def __init__(self, config: SimulationConfig, test_case: Any):
        """
        Initialize the solver with simulation configuration and a specific test case.

        Parameters:
        ----------
        config : SimulationConfig
            The simulation configuration parameters.
        test_case : BaseTestCase
            The specific test case to run.
        """
        self.config = config
        self.test_case = test_case
        self.grid = config.grid
        self.variables = config.variables
        self.boundary_manager = BoundaryConditionManager()
        self.flux = Flux(
            riemann_solver=test_case.config.numerics.riemann_solver,
            limiter=test_case.config.numerics.limiter,
            variables=self.variables,
            grid=self.grid,
        )
        self.time = 0.0
        self.step = 0

    def initialize(self):
        """
        Set up the simulation by initializing grid, variables, and boundary conditions.
        """
        self.grid.initialize(self.config.spatial_grid)
        self.variables.initialize(self.grid)
        self.boundary_manager.setup_conditions(self.test_case.boundary_conditions)
        print("Initialization complete.")

    def run(self):
        """
        Execute the simulation loop until the maximum number of steps or final time is reached.
        """
        self.initialize()
        temporal = self.config.temporal

        while self.step < temporal.stepmax and self.time < temporal.tout[-1]:
            self.step += 1
            self.time += temporal.dtfixed  # Example time increment

            self.apply_boundary_conditions()
            self.compute_fluxes()
            self.update_variables()

            if self.step % 100 == 0:
                self.output_results()

        print("Simulation completed.")

    def apply_boundary_conditions(self):
        """
        Apply boundary conditions to the current solver state.
        """
        solver_state = {
            "time": self.time,
            "step": self.step,
            # Add more state information as needed
        }
        self.boundary_manager.apply_boundary_conditions(
            self.grid.cells, self.grid.faces, solver_state
        )
        print(f"Applied boundary conditions at step {self.step}.")

    def compute_fluxes(self):
        """
        Compute fluxes based on the current state of variables and grid.
        """
        flux_result = self.flux.compute_averaging_fluxes()
        self.variables.update_cell_vars(flux_result)
        print(f"Computed fluxes at step {self.step}.")

    def update_variables(self):
        """
        Update the simulation variables based on computed fluxes and time step.
        """
        dt = self.config.temporal.dtfixed
        self.variables.update_conservative_variables(dt)
        print(f"Updated variables at step {self.step}.")

    def output_results(self):
        """
        Output the simulation results at specified intervals.
        """
        # Implement output logic, e.g., writing to files or plotting
        print(f"Output results at step {self.step}, time {self.time}.")


# def __init__(self, config: SimulationConfig, variables: Variables, flux: Flux, advection, integrator):
#     """
#     Parameters:
#         config     : SimulationConfig instance containing global constants, spatial grid, boundary
#                      conditions, temporal settings, and more. Note that in its __post_init__, the
#                      usable Grid (config.grid) is created.
#         variables  : An instance of Variables holding the cell_vars (shape: (nx, ny, [nz], num_vars)).
#         flux       : A Flux container initialized with a specific Riemann solver, limiter, and reconstruction.
#         advection  : An advection routine (function or object) that exposes a method compute_fluxes(...) to
#                      compute the numerical fluxes given (grid, variables, flux, config).
#         integrator : A time integration routine that exposes an integrate(...) method for advancing cell_vars.
#     """
#     self.config = config
#     self.grid = config.grid  # The configurable grid is built on config creation.
#     self.variables = variables
#     self.flux = flux  # This container already holds the proper riemann solver, limiter, and reconstruction.
#     self.advection = advection
#     self.integrator = integrator
#
#
# def apply_boundary_conditions(self):
#     """
#     Enforce the boundary conditions saved in the simulation configuration.
#
#     Here we assume that self.config.boundary_conditions.conditions is a mapping between
#     BoundarySide enums and BdryType enums. A factory is assumed to provide for a given boundary side
#     and condition the appropriate boundary condition object. That object (with an .apply() method)
#     is then used to update self.variables appropriately.
#     """
#     for side, bc_type in self.config.boundary_conditions.conditions.items():
#         # The get_boundary_condition method is assumed to perform the registry lookup.
#         bc = self.config.boundary_conditions.get_boundary_condition(side, bc_type)
#         bc.apply(side, self.grid, self.variables)
#
#
# def compute_fluxes(self):
#     """
#     Compute the numerical fluxes using the provided advection routine.
#
#     The advection routine is expected to use values already stored in the provided flux container.
#     This ensures that the flux computation will make use of the proper Riemann solver, limiter, and
#     reconstruction algorithm without instantiating a new flux object.
#
#     Returns:
#         fluxes: The computed flux array (or similar data structure) compatible with the later update step.
#     """
#     fluxes = self.advection.compute_fluxes(
#         grid=self.grid,
#         variables=self.variables,
#         flux=self.flux,
#         config=self.config
#     )
#     return fluxes
#
#
# def step(self, dt: float):
#     """
#     Perform one time-step: enforce boundary conditions, compute fluxes using the advection routine,
#     and update the solution using the time integrator.
#
#     Parameters:
#         dt : Time-step increment.
#     """
#     # 1. Update the ghost cells / boundary cells according to the current configuration.
#     self.apply_boundary_conditions()
#
#     # 2. Calculate numerical fluxes given the current cell state.
#     fluxes = self.compute_fluxes()
#
#     # 3. Advance the solution in time; integrator.integrate returns a new cell_vars array.
#     self.variables.cell_vars = self.integrator.integrate(
#         cell_vars=self.variables.cell_vars,
#         fluxes=fluxes,
#         dt=dt,
#         grid=self.grid,
#         config=self.config
#     )
#
#
# def run(self, final_time: float, dt: float):
#     """
#     Execute the simulation loop until final_time is reached.
#
#     This loop applies boundary conditions, calculates fluxes, and updates the cell_vars repeatedly.
#
#     Parameters:
#         final_time: The desired final simulation time.
#         dt        : The time step size.
#
#     Returns:
#         variables: The Variables instance containing the final state after time integration.
#     """
#     current_time = 0.0
#     while current_time < final_time:
#         self.step(dt)
#         current_time += dt
#         # Optionally, insert diagnostic, output, or monitoring callbacks here.
#     return self.variables


###################################3

# class Solver:
#     def init(self, config: SimulationConfig, variables: Variables, flux: Flux,
#         advection, integrator, bc_manager: BoundaryConditionManager):
#         """
#         Parameters:
#         config     : The SimulationConfig instance.
#         variables  : Instance of Variables containing cell_vars.
#         flux       : The Flux container (already configured with riemann solver, limiter,
#         and reconstruction routine).
#         advection  : The external advection routine.
#         integrator : The external time integrator.
#         bc_manager : An instance of BoundaryConditionManager for handling BC setup,
#         validation, and application.
#         """
#             self.config = config
#             self.grid = config.grid
#             self.variables = variables
#             self.flux = flux
#             self.advection = advection
#             self.integrator = integrator
#             self.bc_manager = bc_manager

# def apply_boundary_conditions(self):
#     """
#     Delegates boundary condition enforcement to the BoundaryConditionManager.
#     """
#     self.bc_manager.apply_boundary_conditions(self.variables.cell_vars)
#
# def compute_fluxes(self):
#     fluxes = self.advection.compute_fluxes(
#         grid=self.grid,
#         variables=self.variables,
#         flux=self.flux,
#         config=self.config
#     )
#     return fluxes
#
# def step(self, dt: float):
#     self.apply_boundary_conditions()
#     fluxes = self.compute_fluxes()
#     self.variables.cell_vars = self.integrator.integrate(
#         cell_vars=self.variables.cell_vars,
#         fluxes=fluxes,
#         dt=dt,
#         grid=self.grid,
#         config=self.config
#     )
#
# def run(self, final_time: float, dt: float):
#     current_time = 0.0
#     while current_time < final_time:
#         self.step(dt)
#         current_time += dt
#     return self.variables

class Solver:
    def init(self, grid, variables, flux, boundary_manager, advection_routine, dt, t_end):
        """ Parameters:
            grid - Grid object containing the spatial discretization.
            variables - Variable container (pre-initialized by initial conditions routines).
            flux - Flux object (provides apply_riemann_solver()).
            boundary - Boundary condition handler.
            advection_routine - A callable (from advection.py) that performs the flux-based update.
            dt - Base time step.
            t_end - End time of the simulation. """
        self.grid = grid
        self.variables = variables
        self.flux = flux
        self.boundary = boundary_manager
        self.advection_routine = advection_routine # injected function for advection updates
        self.dt = dt
        self.t_end = t_end

        #
        if self.grid.ndim == 1:
            self.directions = ["x"]
        elif self.grid.ndim == 2:
            self.directions = ["x", "y"]
        elif self.grid.ndim == 3:
            self.directions = ["x", "y", "z"]
        else:
            raise ValueError("Unsupported number of dimensions.")

    def step(self):
        print("=== Beginning time step ===")
        # Apply boundary conditions first.
        self.boundary.apply(self.variables.cell_vars)

        # Apply Strang splitting based on the number of dimensions.
        if len(self.directions) == 1:
            # Single dimension: Perform full update.
            self.advection_routine(self.flux, self.variables, self.grid, self.directions[0], self.dt)
        else:
            # For 2D or 3D: update each spatial direction.
            # First half-steps for all directions except the last:
            for d in self.directions[:-1]:
                self.advection_routine(self.flux, self.variables, self.grid, d, self.dt / 2)
            # Full step for the last direction:
            self.advection_routine(self.flux, self.variables, self.grid, self.directions[-1], self.dt)
            # Reverse half-steps for the remaining directions:
            for d in reversed(self.directions[:-1]):
                self.advection_routine(self.flux, self.variables, self.grid, d, self.dt / 2)
        print("=== End time step ===\n")

    def run(self):
        t = 0.0
        while t < self.t_end:
            print(f"Time: {t}")
            self.step()
            t += self.dt
        print("Simulation finished.")