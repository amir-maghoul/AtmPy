"""Context module for the time integrators"""

from dataclasses import dataclass, field
from typing import Dict, Any, Generic, TYPE_CHECKING

if TYPE_CHECKING:
    from atmpy.grid.kgrid import Grid
    from atmpy.variables.variables import Variables
    from atmpy.flux.flux import Flux
    from atmpy.boundary_conditions.boundary_manager import BoundaryManager
    from atmpy.infrastructure.enums import AdvectionRoutines
from atmpy.infrastructure.enums import TimeIntegrators
from atmpy.infrastructure.factory import get_time_integrator
from atmpy.time_integrators.abstract_time_integrator import TTimeIntegrator


@dataclass
class TimeIntegratorContext(Generic[TTimeIntegrator]):
    """
    A generic instantiation context for time integrators.

    This context collects the common dependencies required by a time integrator,
    including the computational grid, variables, flux object, boundary manager, time step parameters, etc.
    For integrators that need extra parameters (for example, the IMEX integrator additionally requires
    a multiple-pressure-variables object (mpv), a Coriolis operator, pressure solver, thermodynamics, and a scaling parameter Msq),
    these extra dependencies can be provided in the extra_dependencies dictionary.

    Attributes
    ----------
    integrator_type:
        The target time integrator type (as defined in the TimeIntegrators enum).
    grid:
        The computational grid.
    variables:
        The variables container.
    flux:
        The flux computation object.
    boundary_manager:
        The boundary condition manager.
    dt:
        The simulation time step.
    t_end:
        The simulation end time.
    maxstep:
        The maximum allowed time steps.
    extra_dependencies: Additional parameters required by a specific integrator (e.g., mpv, coriolis_operator, pressure_solver, thermodynamics, Msq).
    """

    integrator_type: TimeIntegrators
    grid: "Grid"
    variables: "Variables"
    flux: "Flux"
    boundary_manager: "BoundaryManager"
    advection_routine: "AdvectionRoutines"
    dt: float
    extra_dependencies: Dict[str, Any] = field(default_factory=dict)

    def instantiate(self) -> TTimeIntegrator:
        """
        Instantiates and returns a time integrator of the selected type, combining
        the common dependencies with any extra integrator-specific dependencies.
        """
        dependencies = {
            "grid": self.grid,
            "variables": self.variables,
            "flux": self.flux,
            "boundary_manager": self.boundary_manager,
            "advection_routine": self.advection_routine,
            "dt": self.dt,
        }
        dependencies.update(self.extra_dependencies)
        return get_time_integrator(self.integrator_type, **dependencies)
