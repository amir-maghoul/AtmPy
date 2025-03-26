"""This module contains different time integrators"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from atmpy.flux.flux import Flux
    from atmpy.grid.kgrid import Grid
    from atmpy.boundary_conditions.boundary_manager import BoundaryManager

from atmpy.solver import advection_routines
from atmpy.variables.variables import Variables
from atmpy.infrastructure.enums import AdvectionRoutines
from atmpy.infrastructure.factory import get_advection_routines


class AbstractTimeIntegrator(ABC):
    """Abstract class for different time integrators

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
        grid: "Grid",
        variables: "Variables",
        flux: "Flux",
        boundary_manager: "BoundaryManager",
        advection_routine: AdvectionRoutines,
        dt: float,
        t_end: float,
        maxstep: int,
    ):
        self.grid: "Grid" = grid
        self.variables: "Variables" = variables
        self.flux: "Flux" = flux
        self.boundary: "BoundaryManager" = boundary_manager
        self.advection_routine: callable = get_advection_routines(advection_routine)
        self.dt: float = dt
        self.t_end: float = t_end
        self.maxstep: int = maxstep
        self.ndim: int = self.grid.ndim
        self.current_time: float = 0

    @abstractmethod
    def step(self):
        pass


class TimeIntegrator(AbstractTimeIntegrator):
    """This is the main time integrator of the project. The algorithm is based on the algorithm introduced in
    BK19 paper."""

    def __init__(
        self,
        grid: "Grid",
        variables: "Variables",
        flux: "Flux",
        boundary_manager: "BoundaryManager",
        advection_routine: callable,
        dt: float,
        t_end: float,
        maxstep: int,
    ):
        super().__init__(
            grid,
            variables,
            flux,
            boundary_manager,
            advection_routine,
            dt,
            t_end,
            maxstep,
        )

    def non_advective_explicit_update(self):
        pass

    def non_advective_implicit_update(self):
        pass
