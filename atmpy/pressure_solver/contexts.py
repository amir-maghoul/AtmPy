"""Instantiation contexts for discrete operators and pressure solvers"""

from dataclasses import dataclass
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from atmpy.infrastructure.enums import (
        PressureSolvers,
        DiscreteOperators,
        LinearSolvers,
    )
    from atmpy.pressure_solver.linear_solvers import ILinearSolver
    from atmpy.pressure_solver.pressure_solvers import AbstractPressureSolver
    from atmpy.pressure_solver.discrete_operations import AbstractDiscreteOperator

from atmpy.infrastructure.factory import (
    get_pressure_solver,
    get_discrete_operators,
    get_linear_solver,
)
from atmpy.infrastructure.enums import LinearSolvers


@dataclass
class DCInstantiationContext:
    """Discrete Operators instantiation context"""

    ndim: int
    dxyz: List[float]


class PSInstantiationContext:
    """Pressure solvers instantiation context. This context is responsible for creating a pressure solver instance.
    It requires a discrete operator dependency."""

    def __init__(
        self,
        solver_type: "PressureSolvers",
        discrete_operator_type: "DiscreteOperators",
        op_context: DCInstantiationContext,
        linear_solver_type: "LinearSolvers",
    ):
        self.solver_type = solver_type
        self.discrete_operator_type = discrete_operator_type
        self.op_context = (
            op_context  # instance of the DCInstantiationContext (e.g., ndim and dxyz)
        )
        self.linear_solver: "ILinearSolver" = get_linear_solver(linear_solver_type)

    def instantiate(self):
        """Create an instance of pressure solver using the given context and the corresponding factory function"""
        # First, create the discrete operator instance using its own context.
        discrete_operator = get_discrete_operators(
            self.discrete_operator_type,
            ndim=self.op_context.ndim,
            dxyz=self.op_context.dxyz,
        )
        # Now create and return the pressure solver instance, passing the discrete operator and linear solver.
        return get_pressure_solver(
            self.solver_type,
            discrete_operator=discrete_operator,
            linear_solver=self.linear_solver,
        )


def example_usage():
    from atmpy.infrastructure.enums import PressureSolvers, DiscreteOperators

    op_context = DCInstantiationContext(ndim=2, dxyz=[0.1, 0.1])
    linear_solver = LinearSolvers.BICGSTAB

    # Instantiate the pressure solver context by specifying enums for pressure solver and discrete operator.
    ps_context = PSInstantiationContext(
        solver_type=PressureSolvers.CLASSIC_PRESSURE_SOLVER,
        discrete_operator_type=DiscreteOperators.CLASSIC_OPERATOR,
        op_context=op_context,
        linear_solver_type=linear_solver,
    )
    pressure_solver_instance = ps_context.instantiate()
    print(pressure_solver_instance.linear_solver)
    print(pressure_solver_instance.discrete_operator)


if __name__ == "__main__":
    example_usage()
