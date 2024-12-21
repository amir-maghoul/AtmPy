""" Module for the definition of the equation of state. Any different and general forms of eos
must be implemented here as a class and then passed to the riemann solvers in the flux. """

from abc import ABC, abstractmethod
import numpy as np
from atmpy.physics.utility import P_to_pressure_numba, exner_to_pressure_numba, exner_sound_speed_numba

class EOS(ABC):
    """ A common interface for any EOS. Each concrete class must define
    how to get pressure and sound speed at minimum."""
    @abstractmethod
    def pressure(self, *args, **kwargs) -> np.ndarray:
        """Compute pressure given density and internal energy."""
        pass

    @abstractmethod
    def sound_speed(self, *args, **kwargs) -> np.ndarray:
        """Compute sound speed given density and pressure."""
        pass

class IdealGasEOS(EOS):
    """ The conventional ideal-gas EOS with gamma. """
    def __init__(self, gamma: float=1.4):
        self.gamma = gamma

    def pressure(self, rho: np.ndarray, e: np.ndarray):
        # p = (gamma - 1) * rho * e
        return (self.gamma - 1.0) * rho * e

    def sound_speed(self, rho: np.ndarray, p: np.ndarray):
        # a = sqrt(gamma * p / rho)
        return np.sqrt(self.gamma * p / rho)


class BarotropicEOS(EOS):
    """ The Barotropic EOS class. """
    def __init__(self, K: float=1.0, gamma: float=1.4):
        self.K = K
        self.gamma = gamma

    def pressure(self, rho: np.ndarray, e: np.ndarray=None):
        return self.K * rho ** self.gamma

    def sound_speed(self, rho: np.ndarray, p: np.ndarray):
        return np.sqrt(self.gamma * p / rho)

class ExnerBasedEOS(EOS):
    """
    Demonstrates an EOS using Exner-based relationships, but leveraging
    Numba-friendly free functions for performance.
    """
    def __init__(self, R, cp, cv, p_ref):
        self.R = R
        self.cp = cp
        self.cv = cv
        self.p_ref = p_ref

    def pressure(self, P: np.ndarray=None, pi: np.ndarray=None):
        """
        Examples of usage:
          - If pi_arr is provided, compute p from pi.
          - If P_arr is provided, compute p from P.
          (In reality, you might pass both or keep separate methods.)
        """
        if P is not None:
            return P_to_pressure_numba(P, self.R, self.p_ref, self.cp, self.cv)
        elif pi is not None:
            return exner_to_pressure_numba(pi, self.p_ref, self.cp, self.R)
        else:
            raise ValueError("Must provide either 'P_arr' or 'pi_arr' to compute pressure.")

    def sound_speed(self, rho_arr: np.ndarray, p_arr: np.ndarray):
        """
        Speed of sound can be approximated via gamma = cp/cv, or a more
        specialized formula. Here we use c = sqrt(gamma * p / rho).
        """
        gamma = self.cp / self.cv
        return exner_sound_speed_numba(rho_arr, p_arr, gamma)
