"""Instantiation contexts for discrete operators and pressure solvers"""

from dataclasses import dataclass, field
from typing import List, TYPE_CHECKING, Generic, Dict, Any, Tuple

if TYPE_CHECKING:
    from atmpy.infrastructure.enums import (
        PressureSolvers,
        DiscreteOperators,
        LinearSolvers,
    )
    from atmpy.grid.kgrid import Grid
from atmpy.pressure_solver.abstract_pressure_solver import TPressureSolver
from atmpy.pressure_solver.discrete_operations import TDiscreteOperator

from atmpy.infrastructure.factory import (
    get_pressure_solver,
    get_discrete_operators,
    get_linear_solver,
)
from atmpy.infrastructure.enums import LinearSolvers


@dataclass
class DiscreteOperatorsContext:
    """Discrete Operators instantiation context"""

    operator_type: "DiscreteOperators"
    grid: "Grid"

    def instantiate(self) -> TDiscreteOperator:
        """Instantiate a discrete operator"""
        return get_discrete_operators(name=self.operator_type, grid=self.grid)


@dataclass
class PressureContext(Generic[TPressureSolver]):
    """Pressure solvers instantiation context. This context is responsible for creating a pressure solver instance.
    It requires a discrete operator dependency."""

    solver_type: "PressureSolvers"
    op_context: DiscreteOperatorsContext  # instance of the DiscreteOperatorContext
    linear_solver_type: "LinearSolvers"
    extra_dependencies: Dict[str, Any] = field(default_factory=dict)

    def instantiate(self) -> TPressureSolver:
        """Create an instance of pressure solver using the given context and the corresponding factory function"""
        # First, create the discrete operator instance using its own context.
        discrete_operator = self.op_context.instantiate()
        linear_solver = get_linear_solver(self.linear_solver_type)
        dependencies = {
            "discrete_operator": discrete_operator,
            "linear_solver": linear_solver,
        }
        dependencies.update(self.extra_dependencies)
        # Now create and return the pressure solver instance, passing the discrete operator and linear solver.
        return get_pressure_solver(self.solver_type, **dependencies)

