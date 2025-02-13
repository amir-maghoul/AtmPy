""" Module for boundary conditions handling"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List


class BaseBoundaryCondition(ABC):
    """Abstract base class for all boundary conditions."""

    def __init__(self, **kwargs):
        self.params = kwargs

    @abstractmethod
    def apply(self, cells, faces, solver_state):
        pass


class SlipWall(BaseBoundaryCondition):
    def apply(self, cells, faces, solver_state):
        # Zero normal velocity, reflect tangential components
        pass


class NonReflectiveOutlet(BaseBoundaryCondition):
    def apply(self, cells, faces, solver_state):
        # Characteristic-based extrapolation
        pass


class WallBoundary(BaseBoundaryCondition):
    def apply(self, cell_vars, grid, direction=None):
        # Implement wall BC logic (e.g., reflect normal velocity)
        pass


class InflowBoundary(BaseBoundaryCondition):
    def apply(self, cell_vars, grid, direction=None):
        # Implement inflow BC logic (e.g., fixed pressure, temperature, velocity)
        pass


class OutflowBoundary(BaseBoundaryCondition):
    def apply(self, cell_vars, grid, direction=None):
        # Implement outflow BC logic (e.g., zero gradient)
        pass


class PeriodicBoundary(BaseBoundaryCondition):
    def apply(self, cell_vars, grid, direction=None):
        # Implement periodic BC logic (copy infrastructure from opposite boundary)
        pass
