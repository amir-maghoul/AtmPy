""" Internal module to test whether a given name exists in the function registry. """

# factories.py

from atmpy.infrastructure.enums import (
    SlopeLimiters,
    RiemannSolvers,
    FluxReconstructions,
    BoundaryConditions,
)
from atmpy.infrastructure.registries import (
    SLOPE_LIMITERS,
    RIEMANN_SOLVERS,
    FLUX_RECONSTRUCTION,
    BOUNDARY_CONDITIONS,
)


def get_slope_limiter(name: SlopeLimiters) -> callable:
    """

    Parameters
    ----------
    name: SlopeLimiters
        The enum member specifying the desired slope limiter.

    Returns
    -------
        Callable: The corresponding slope limiter function.

    """
    try:
        return SLOPE_LIMITERS[name]
    except KeyError:
        raise ValueError(f"Unknown slope limiter: {name}")


def get_riemann_solver(name: RiemannSolvers) -> callable:
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

    try:
        return RIEMANN_SOLVERS[name]
    except KeyError:
        raise ValueError(f"Unknown Riemann solver: {name}")


def get_reconstruction_method(name: FluxReconstructions) -> callable:
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
    try:
        return FLUX_RECONSTRUCTION[name]
    except KeyError:
        raise ValueError(f"Unknown reconstruction method: {name}")


def get_boundary_conditions(name: BoundaryConditions, **params) -> object:
    """
    Retrieves the boundary condition class based on the provided enum member.

    Parameters
    ----------
    name: BoundaryConditions
        The enum member specifying the desired boundary condition.
    params: dict
        The parameters for the boundary condition class.

    Returns
    -------
    BoundaryCondition: The corresponding boundary condition class.

    """
    try:
        boundary_condition_class = BOUNDARY_CONDITIONS[name]
        return boundary_condition_class(**params)
    except KeyError:
        raise ValueError(f"Unknown boundary conditions: {name}")
