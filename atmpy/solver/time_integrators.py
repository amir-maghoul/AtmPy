"""This module contains different time integrators"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from atmpy.flux.flux import Flux
    from atmpy.grid.kgrid import Grid
    from atmpy.boundary_conditions.boundary_manager import BoundaryManager
    from atmpy.physics.thermodynamics import Thermodynamics
    from atmpy.variables.multiple_pressure_variables import MPV

from atmpy.solver import advection_routines
from atmpy.variables.variables import Variables
from atmpy.infrastructure.enums import AdvectionRoutines
from atmpy.infrastructure.factory import get_advection_routines
from atmpy.infrastructure.enums import VariableIndices as VI
from atmpy.solver.discrete_functions import nodal_gradient


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
    th: Thermodynamics,
        The dataclass containing the thermodynamic constants.
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
        mpv: "MPV",
        flux: "Flux",
        boundary_manager: "BoundaryManager",
        advection_routine: AdvectionRoutines,
        th: "Thermodynamics",
        dt: float,
        t_end: float,
        maxstep: int,
    ):
        self.grid: "Grid" = grid
        self.variables: "Variables" = variables
        self.mpv: "MPV" = mpv
        self.flux: "Flux" = flux
        self.boundary: "BoundaryManager" = boundary_manager
        self.advection_routine: callable = get_advection_routines(advection_routine)
        self.th: "Thermodynamics" = th
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
        mpv: "MPV",
        flux: "Flux",
        boundary_manager: "BoundaryManager",
        advection_routine: callable,
        th: "Thermodynamics",
        dt: float,
        t_end: float,
        maxstep: int,
    ):
        super().__init__(
            grid,
            variables,
            mpv,
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

    def nodal_correction(self, p: np.ndarray, direction: str):
        """ Apply a correction on the values of nodal variables, specifically on pressure variables.

        Parameters
        ----------
        direction: str
            The direction of gravity/stratification
        """

        Gammainv = self.th.Gammainv

        # Compute the derivative of Chi in the direction of gravity/stratification
        dS = self.mpv.compute_dS_on_nodes(direction)

        # Compute the gradient of the pressure variable
        Dpx, Dpy, Dpz = nodal_gradient(p, self.ndim, self.grid.dxyz)

        # intermediate variable for Theta
        Y = self.variables.cell_vars[..., VI.RHOY] / self.variables.cell_vars[..., VI.RHO]
        coeff = (Gammainv * self.variables.cell_vars[..., VI.RHOY] * Y)

        self.mpv.u = -self.dt * coeff * Dpx
        self.mpv.v = -self.dt * coeff * Dpy
        self.mpv.w = -self.dt * coeff * Dpz

        multiply_inverse_coriolis(mpv, Sol, mpv, ud, elem, elem, dt, attrs=['u', 'v', 'w'])

        # intermediate variable for Chi.
        chi = 1.0 / Y

        Sol.rhou += chi * mpv.u
        Sol.rhov += chi * mpv.v
        Sol.rhow += chi * mpv.w if ndim == 3 else 0.0
        Sol.rhoX += - updt_chi * dt * dSdy * Sol.rhov

        # set_explicit_boundary_data(Sol, elem, ud, th, mpv)

        assert True
