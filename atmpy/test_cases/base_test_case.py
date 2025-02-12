from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseTestCase(ABC):
    def __init__(self, name: str):
        self.name = name
        self.parameters: Dict[str, Any] = {}
        self.initial_conditions: Dict[str, Any] = {}
        self.boundary_conditions: Dict[str, Dict[str, Any]] = {}

    @abstractmethod
    def setup(self):
        """Initialize the test case with specific parameters, initial conditions, and boundary conditions."""
        pass
