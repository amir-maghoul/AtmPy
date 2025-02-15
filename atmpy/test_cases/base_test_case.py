""" Module for abstract base class for test cases."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
from atmpy.configuration.simulation_configuration import (
    SimulationConfig,
)
from atmpy.configuration.simulation_data import BoundaryFace
from atmpy.infrastructure.enums import BoundarySide, BoundaryConditions as BdryType


class BaseTestCase(ABC):
    def __init__(self, name: str, config: SimulationConfig):
        self.name = name
        self.config = config
        self.parameters: Dict[str, Any] = {}
        self.initial_conditions: Dict[str, Any] = {}
        self.boundary_conditions: Dict[BoundarySide, BdryType] = {}

    def set_boundary_condition(
        self,
        boundary_side: BoundarySide,
        condition: BdryType,
        params: Dict[str, Any],
        faces: List[BoundaryFace],
    ):
        self.config.update_boundary_condition(boundary_side, condition)
        self.boundary_conditions[boundary_side] = condition

    @abstractmethod
    def setup(self):
        """Initialize the test case with specific parameters, initial conditions, and boundary conditions."""
        pass
