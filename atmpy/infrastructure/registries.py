"""This is a registry container to map names to their corresponding functions."""

from atmpy.flux.riemann_solvers import *
from atmpy.flux.reconstruction import *
from atmpy.flux.limiters import *
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
    Preconditioners,
)
from atmpy.boundary_conditions.boundary_conditions import *
from atmpy.time_integrators.advection_routines import *
from atmpy.pressure_solver.linear_solvers import *
from atmpy.time_integrators.imex_operator_splitting import IMEXTimeIntegrator
from atmpy.pressure_solver.discrete_operations import *
from atmpy.pressure_solver.preconditioners import *
from atmpy.pressure_solver.classical_pressure_solvers import ClassicalPressureSolver

SLOPE_LIMITERS = {
    SlopeLimiters.MINMOD: minmod,
    SlopeLimiters.VAN_LEER: van_leer,
    SlopeLimiters.SUPERBEE: superbee,
    SlopeLimiters.MC_LIMITER: mc_limiter,
    SlopeLimiters.AVERAGE: average,
    SlopeLimiters.OSPRE: ospre,
    SlopeLimiters.KOREN: koren,
}

RIEMANN_SOLVERS = {
    RiemannSolvers.RUSANOV: rusanov,
    RiemannSolvers.MODIFIED_HLL: modified_hll,
    RiemannSolvers.HLL: hll,
    RiemannSolvers.HLLC: hllc,
    RiemannSolvers.ROE: roe,
}

FLUX_RECONSTRUCTION = {
    FluxReconstructions.MODIFIED_MUSCL: modified_muscl,
    FluxReconstructions.PIECEWISE_CONSTANT: piecewise_constant,
    FluxReconstructions.MUSCL: muscl,
}

BOUNDARY_CONDITIONS = {
    BoundaryConditions.WALL: Wall,
    BoundaryConditions.INFLOW: InflowBoundary,
    BoundaryConditions.OUTFLOW: OutflowBoundary,
    BoundaryConditions.NON_REFLECTIVE_OUTLET: NonReflectiveOutlet,
    BoundaryConditions.PERIODIC: PeriodicBoundary,
    BoundaryConditions.REFLECTIVE_GRAVITY: ReflectiveGravityBoundary,
}

ADVECTION_ROUTINES = {
    AdvectionRoutines.STRANG_SPLIT: upwind_strang_split_advection,
    AdvectionRoutines.FIRST_ORDER_RK: first_order_splitting_advection,
}

LINEAR_SOLVERS = {
    LinearSolvers.BICGSTAB: BiCGStabSolver,
    LinearSolvers.GMRES: GMRESSolver,
}

TIME_INTEGRATORS = {
    TimeIntegrators.IMEX: IMEXTimeIntegrator,
}

DISCRETE_OPERATORS = {DiscreteOperators.CLASSIC_OPERATOR: ClassicalDiscreteOperator}

PRESSURE_SOLVERS = {
    PressureSolvers.CLASSIC_PRESSURE_SOLVER: ClassicalPressureSolver,
}

PRECONDITIONERS = {
    Preconditioners.DIAGONAL: apply_inverse_diagonal,
    Preconditioners.COLUMN: apply_inverse_tridiagonal,
}

PRECONDITIONER_COMPONENTS = {
    Preconditioners.DIAGONAL: compute_inverse_diagonal_components,
    Preconditioners.COLUMN: compute_tridiagonal_components,
}
