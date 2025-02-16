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
        self.grid = Grid()
        self.variables = Variables()
        self.boundary_manager = BoundaryConditionManager()
        self.flux = Flux(
            riemann_solver=test_case.config.numerics.riemann_solver,
            limiter=test_case.config.numerics.limiter,
            variables=self.variables,
            grid=self.grid
        )
        self.time = 0.0
        self.step = 0

    def initialize(self):
        """
        Set up the simulation by initializing grid, variables, and boundary conditions.
        """
        self.grid.initialize(self.config.spatial_grid)
        self.variables.initialize(self.grid)
        self.boundary_manager.setup_conditions(
            self.test_case.boundary_conditions
        )
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
            "step": self.step
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
