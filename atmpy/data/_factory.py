""" Internal module to test whether a given name exists in the function registry. """

# factories.py

from atmpy.data.enums import SlopeLimiters, RiemannSolvers, FluxReconstructions, BoundaryConditions
from atmpy.data.registries import SLOPE_LIMITERS, RIEMANN_SOLVERS, FLUX_RECONSTRUCTION, BOUNDARY_CONDITIONS



def get_slope_limiter(name: SlopeLimiters):
    try:
        return SLOPE_LIMITERS[name]
    except KeyError:
        raise ValueError(f"Unknown slope limiter: {name}")


def get_riemann_solver(name: RiemannSolvers):
    try:
        return RIEMANN_SOLVERS[name]
    except KeyError:
        raise ValueError(f"Unknown Riemann solver: {name}")


def get_reconstruction_method(name: FluxReconstructions):
    try:
        return FLUX_RECONSTRUCTION[name]
    except KeyError:
        raise ValueError(f"Unknown reconstruction method: {name}")

def get_boundary_conditions(name: BoundaryConditions):
    try:
        return BOUNDARY_CONDITIONS[name]
    except KeyError:
        raise ValueError(f"Unknown boundary conditions: {name}")
