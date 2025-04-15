""" This module contains the abstract class for the pressure solvers."""

from abc import ABC
from typing import TypeVar

from atmpy.physics.thermodynamics import Thermodynamics
from atmpy.pressure_solver.discrete_operations import AbstractDiscreteOperator
from atmpy.pressure_solver.linear_solvers import ILinearSolver
from atmpy.time_integrators.coriolis import CoriolisOperator


class AbstractPressureSolver(ABC):
    """Create different definitions of pressure solvers."""

    def __init__(
        self,
        discrete_operator: "AbstractDiscreteOperator",
        linear_solver: "ILinearSolver",
        coriolis: "CoriolisOperator",
        thermodynamics: "Thermodynamics",
        Msq: float,
        dt: float,
    ):
        self.discrete_operator: "AbstractDiscreteOperator" = discrete_operator
        self.linear_solver: "ILinearSolver" = linear_solver
        self.coriolis: "CoriolisOperator" = coriolis
        self.th: "Thermodynamics" = thermodynamics
        self.Msq: float = Msq
        self.dt: float = dt
        self.vertical_momentum_index: int = self.coriolis.gravity.gravity_momentum_index


TPressureSolver = TypeVar('TPressureSolver', bound=AbstractPressureSolver)
