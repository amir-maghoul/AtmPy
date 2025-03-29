"""This module includes the abstract structure for the time integrator classes"""

from abc import ABC, abstractmethod


class AbstractTimeIntegrator(ABC):
    """Abstract class for time integrators"""

    @abstractmethod
    def step(self):
        """Advance the integration on time step"""
        raise NotImplementedError("step() must be implemented by subclasses")

    @abstractmethod
    def get_dt(self):
        """Get the timestep size. Used in going to the next time step in the solver class."""
        raise NotImplementedError("get_dt() must be implemented by subclasses")
