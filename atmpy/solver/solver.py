# solver.py
import numpy as np
from typing import Any
from atmpy.boundary_conditions.boundary_manager import BoundaryManager
from atmpy.flux.flux import Flux
from atmpy.grid.kgrid import Grid
from atmpy.variables.variables import Variables

class Solver:
    """The manager class for connecting other parts of the code together.

    Attributes:
    ----------
    grid: Grid
        Object containing the spatial discretization.
    variables: Variables
        Variable container, already initialized elsewhere
    flux: Flux
        Flux object.
    boundary: BoundaryManager
        The boundary condition handler
    advection_routine: Callable
        The advection routine as a function
    time_integrator: Callable
        The time integrator as a function
    dt: float
        The time step size
    t_end: float
        The end time
    current_time: float
        The current timestep of the simulation
    """

    def __init__(
        self,
        grid,
        variables,
        flux,
        BoundaryManager,
        advection_routine,
        time_integrator,
        dt,
        t_end,
        maxstep,
    ):
        """
        Parameters:
        ----------
        grid: Grid
            Object containing the spatial discretization.
        variables: Variables
            Variable container, already initialized elsewhere
        flux: Flux
            Flux object.
        boundary: BoundaryManager
            The boundary condition handler
        advection_routine: AdvectionRoutine (Enum)
            The enum based name of the advection routine
        time_integrator: TimeIntegrator (Enum)
            The enum based name of the time integrator
        dt: float
            Base time step.
        t_end: float
            End time of the simulation."""
        self.grid = grid
        self.variables = variables
        self.flux = flux
        self.boundary = boundary_manager
        self.advection_routine = get_advection_routine(advection_routine)
        self.time_integrator = get_time_integrators(time_integrator)
        self.dt = dt
        self.t_end = t_end
        self.maxstep = maxstep
        self.current_time = 0

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
            self.advection_routine(
                self.flux, self.variables, self.grid, self.directions[0], self.dt
            )
        else:
            # For 2D or 3D: update each spatial direction.
            # First half-steps for all directions except the last:
            for d in self.directions[:-1]:
                self.advection_routine(
                    self.flux, self.variables, self.grid, d, self.dt / 2
                )
            # Full step for the last direction:
            self.advection_routine(
                self.flux, self.variables, self.grid, self.directions[-1], self.dt
            )
            # Reverse half-steps for the remaining directions:
            for d in reversed(self.directions[:-1]):
                self.advection_routine(
                    self.flux, self.variables, self.grid, d, self.dt / 2
                )
        print("=== End time step ===\n")

    def run(self):
        t = 0.0
        while t < self.t_end and step < self.maxstep:
            print(f"Time: {t}")
            self.step()
            t += self.dt
            step += 1
        print("Simulation finished.")
