""" Module for the definition of the equation of state. Any different and general forms of eos
must be implemented here as a class and then passed to the riemann solvers in the flux. """
from abc import ABC, abstractmethod
import numpy as np

class EOS(ABC):
    @abstractmethod
    def pressure(self, rho, e=None):
        """Compute pressure given density and internal energy."""
        pass

    @abstractmethod
    def sound_speed(self, rho, p):
        """Compute sound speed given density and pressure."""
        pass

class IdealGasEOS(EOS):
    def __init__(self, gamma=1.4):
        self.gamma = gamma

    def pressure(self, rho, e=None):
        if e is not None:
            return (self.gamma - 1.0) * rho * e
        else:
            # For compatibility if e is not provided
            raise ValueError("Internal energy 'e' must be provided for IdealGasEOS.")

    def sound_speed(self, rho, p):
        return np.sqrt(self.gamma * p / rho)

class BarotropicEOS(EOS):
    def __init__(self, K=1.0, gamma=1.4):
        self.K = K
        self.gamma = gamma

    def pressure(self, rho, e=None):
        return self.K * rho ** self.gamma

    def sound_speed(self, rho, p):
        return np.sqrt(self.gamma * p / rho)