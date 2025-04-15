"""Internal module to test whether a given name exists in the function registry."""

# factories.py

from atmpy.infrastructure.registries import (
    SLOPE_LIMITERS,
    RIEMANN_SOLVERS,
    FLUX_RECONSTRUCTION,
    BOUNDARY_CONDITIONS,
    ADVECTION_ROUTINES,
    LINEAR_SOLVERS,
    TIME_INTEGRATORS,
    DISCRETE_OPERATORS,
    PRESSURE_SOLVERS,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from atmpy.boundary_conditions.contexts import (
        BaseBoundaryCondition,
        BCInstantiationOptions,
    )
    from atmpy.pressure_solver.linear_solvers import ILinearSolver
    from atmpy.time_integrators.abstract_time_integrator import AbstractTimeIntegrator
    from atmpy.pressure_solver.abstract_pressure_solver import AbstractPressureSolver
    from atmpy.pressure_solver.discrete_operations import AbstractDiscreteOperator
    from atmpy.infrastructure.enums import (
        SlopeLimiters,
        RiemannSolvers,
        FluxReconstructions,
        BoundaryConditions,
        AdvectionRoutines,
        LinearSolvers,
        TimeIntegrators,
        DiscreteOperators,
        PressureSolvers,
    )


def get_slope_limiter(name: "SlopeLimiters") -> callable:
    """

    Parameters
    ----------
    name: SlopeLimiters
        The enum member specifying the desired slope limiter.

    Returns
    -------
        Callable: The corresponding slope limiter function.

    """
    slope_limiter = SLOPE_LIMITERS.get(name)

    if slope_limiter is None:
        raise ValueError(f"Unknown slope limiter: {name}")

    return slope_limiter


def get_riemann_solver(name: "RiemannSolvers") -> callable:
    """
    Retrieves the Riemann solver function based on the provided enum member.

    Parameters:
    ----------
    name: RiemannSolvers
        The enum member specifying the desired Riemann solver.

    Returns:
    --------
        Callable: The corresponding Riemann solver function.

    """

    riemann_solver = RIEMANN_SOLVERS.get(name)

    if riemann_solver is None:
        raise ValueError(f"Unknown Riemann solver: {name}")

    return riemann_solver


def get_reconstruction_method(name: "FluxReconstructions") -> callable:
    """
    Retrieves the flux reconstruction method based on the provided enum member.

    Parameters
    ----------
    name: FluxReconstructions
        The enum member specifying the desired flux reconstruction method.

    Returns
    -------
    Callable: The corresponding flux reconstruction method.
    """

    flux_reconstruction = FLUX_RECONSTRUCTION.get(name)

    if flux_reconstruction is None:
        raise ValueError(f"Unknown reconstruction method: {name}")

    return flux_reconstruction


def get_boundary_conditions(
    name: "BoundaryConditions", context: "BCInstantiationOptions"
) -> "BaseBoundaryCondition":
    """
    Retrieves the boundary condition class based on the provided enum member.

    Parameters
    ----------
    name: BoundaryConditions (enum)
        The enum member specifying the desired boundary condition.
    context: BCInstantiationOptions
        The context object containing the boundary condition configuration.

    Returns
    -------
    BaseBoundaryCondition: The corresponding boundary condition class.

    """
    boundary_condition_class = BOUNDARY_CONDITIONS.get(name)
    if boundary_condition_class is None:
        raise ValueError(f"Unknown boundary conditions: {name}")

    return boundary_condition_class(context)


def get_advection_routines(name: "AdvectionRoutines") -> callable:
    """
    Retrieves the advection routine based on the provided enum member.

    Parameters
    ----------
    name: AdvectionRoutines (enum)
        The enum member specifying the desired advection routine.

    Returns
    -------
    Callable: The corresponding advection function.
    """
    advection_routine = ADVECTION_ROUTINES.get(name)

    if advection_routine is None:
        raise ValueError(f"Unknown advection routine: {name}")

    return advection_routine


def get_linear_solver(name: "LinearSolvers") -> "ILinearSolver":
    """
    Retrieves the linear solver class based on the provided enum member.

    Parameters
    ----------
    name: LinearSolvers (enum)
        The enum member specifying the desired linear solver.

    Returns
    -------
    ILinearSolver: The corresponding linear solver class.

    """
    linear_solver_class = LINEAR_SOLVERS.get(name)

    if linear_solver_class is None:
        raise ValueError(f"Unknown linear solver: {name}")

    return linear_solver_class()


def get_discrete_operators(
    name: "DiscreteOperators", **dependencies
) -> "AbstractDiscreteOperator":
    """
    Retrieves the discrete operator class based on the provided enum member.

    Parameters
    ----------
    name: DiscreteOperators (enum)
        The enum member specifying the desired discrete operator.

    Returns
    -------
    AbstractDiscreteOperator: The corresponding discrete operator class.

    """
    discrete_operator_class = DISCRETE_OPERATORS.get(name)

    if discrete_operator_class is None:
        raise ValueError(f"Unknown discrete operator type: {name}")

    return discrete_operator_class(**dependencies)


def get_pressure_solver(
    name: "PressureSolvers", **dependencies
) -> "AbstractPressureSolver":
    """
    Retrieves the pressure solver class based on the provided enum member.

    Parameters
    ----------
    name: PressureSolvers
        The enum member specifying the desired pressure solver.

    Returns
    -------
    AbstractPressureSolver: The corresponding pressure solver class.

    """
    pressure_solver_class = PRESSURE_SOLVERS.get(name)

    if pressure_solver_class is None:
        raise ValueError(f"Unknown pressure solver type: {name}")

    return pressure_solver_class(**dependencies)


def get_time_integrator(
    name: "TimeIntegrators", **dependencies
) -> "AbstractTimeIntegrator":
    """
    Retrieves the time integrator class based on the provided enum member.

    Parameters
    ----------
    name: TimeIntegrators (enum)
        The enum member specifying the desired time integrator.

    Returns
    -------
    AbstractTimeIntegrator: The corresponding time integrator class.

    """
    integrator_class = TIME_INTEGRATORS.get(name)

    if integrator_class is None:
        raise ValueError(f"Unknown time integrator type: {name}")

    return integrator_class(**dependencies)
