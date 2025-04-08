"""Context module for the time integrators"""

from dataclasses import dataclass, field
from typing import Dict, Any

# Import core dependency types (adjust the import paths as needed)
from atmpy.grid.kgrid import Grid
from atmpy.variables.variables import Variables
from atmpy.flux.flux import Flux
from atmpy.boundary_conditions.boundary_manager import BoundaryManager
from atmpy.infrastructure.enums import TimeIntegrators
from atmpy.infrastructure.factory import get_time_integrator
from atmpy.time_integrators.abstract_time_integrator import AbstractTimeIntegrator


@dataclass
class TimeIntegratorContext:
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
    grid: Grid
    variables: Variables
    flux: Flux
    boundary_manager: BoundaryManager
    dt: float
    extra_dependencies: Dict[str, Any] = field(default_factory=dict)

    def instantiate(self) -> AbstractTimeIntegrator:
        """
        Instantiates and returns a time integrator of the selected type, combining
        the common dependencies with any extra integrator-specific dependencies.
        """
        dependencies = {
            "grid": self.grid,
            "variables": self.variables,
            "flux": self.flux,
            "boundary_manager": self.boundary_manager,
            "dt": self.dt,
        }
        # Merge in any extra dependencies required by the specific integrator.
        dependencies.update(self.extra_dependencies)
        # Use the factory to get the time integrator instance.
        return get_time_integrator(self.integrator_type, **dependencies)


# Example usage:
# For an IMEX integrator (which in imex_operator_splitting.py citeturn1file1 requires extra parameters):
#
# context = TimeIntegratorContext(
#     integrator_type=TimeIntegrators.IMEX,
#     grid=grid,
#     variables=variables,
#     flux=flux,
#     boundary_manager=manager,
#     dt=0.1,
#     extra_dependencies={
#         "mpv": mpv,
#         "coriolis_operator": coriolis,
#         "pressure_solver": pressure,
#         "thermodynamics": th,
#         "Msq": 1.0,
#         "wind_speed": [0.0, 0.0, 0.0]  # optional: override default wind speed
#     }
# )
#
# time_integrator = context.instantiate()
# time_integrator.step()  # Advance the simulation by one time step
