""" This module contains the abstract class for the pressure solvers."""

from abc import ABC, abstractmethod
from typing import TypeVar, TYPE_CHECKING, Callable


if TYPE_CHECKING:
    from atmpy.physics.thermodynamics import Thermodynamics
    from atmpy.pressure_solver.discrete_operations import AbstractDiscreteOperator
    from atmpy.pressure_solver.linear_solvers import ILinearSolver
    from atmpy.time_integrators.coriolis import CoriolisOperator
    from atmpy.infrastructure.enums import Preconditioners


class AbstractPressureSolver(ABC):
    """Create different definitions of pressure solvers."""

    def __init__(
        self,
        discrete_operator: "AbstractDiscreteOperator",
        linear_solver: "ILinearSolver",
        precondition_type: "Preconditioners",
        coriolis: "CoriolisOperator",
        thermodynamics: "Thermodynamics",
        Msq: float,
    ):
        self.discrete_operator: "AbstractDiscreteOperator" = discrete_operator
        self.linear_solver: "ILinearSolver" = linear_solver
        self.coriolis: "CoriolisOperator" = coriolis
        self.th: "Thermodynamics" = thermodynamics
        self.Msq: float = Msq
        self.vertical_momentum_index: int = self.coriolis.gravity.gravity_momentum_index
        self.precondition_type: "Preconditioners" = precondition_type
        self.precon_apply: Callable


TPressureSolver = TypeVar("TPressureSolver", bound=AbstractPressureSolver)
