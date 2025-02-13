""" Module for the definition of the equation of state. Any different and general forms of eos
must be implemented here as a class and then passed to the riemann solvers in the flux. """

from abc import ABC, abstractmethod
import numpy as np
from atmpy.physics.utility import (
    P_to_pressure_numba,
    exner_to_pressure_numba,
    exner_sound_speed_numba,
)


class EOS(ABC):
    """A common interface for any EOS. Each concrete class must define
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
    """The conventional ideal-gas EOS with gamma."""

    def __init__(self, gamma: float = 1.4):
        self.gamma = gamma

    def pressure(self, *args, **kwargs):
        rho = args[0]
        e = args[1]
        return (self.gamma - 1.0) * rho * e

    def sound_speed(self, *args, **kwargs):
        rho = args[0]
        p = args[1]
        # a = sqrt(gamma * p / rho)
        return np.sqrt(self.gamma * p / rho)


class BarotropicEOS(EOS):
    """The Barotropic EOS class."""

    def __init__(self, K: float = 1.0, gamma: float = 1.4):
        self.K = K
        self.gamma = gamma

    def pressure(self, *args, **kwargs):
        rho = args[0]
        return self.K * rho**self.gamma

    def sound_speed(self, *args, **kwargs):
        rho = args[0]
        p = args[1]
        return np.sqrt(self.gamma * p / rho)


class ExnerBasedEOS(EOS):
    """
    Demonstrates an EOS using Exner-based relationships, but leveraging
    Numba-friendly free functions for performance.
    """

    def __init__(self, cp: float = 1000, cv: float = 750, p_ref: float = 1.0):
        """Constructor.

        Attributes
        ----------
        p_ref : float
            the reference pressure
        cp : float
            the heat capacity in constant pressure
        cv : float
            the heat capacity in constant volume
        R : float
            Specific gas constant for dry air
        """
        self.cp = cp
        self.cv = cv
        self.R = cp - cv
        self.p_ref = p_ref
        self.gamma = self.cp / self.cv

    def pressure(self, *args, **kwargs):
        """
        Calculate pressure given unphysical pressure P or Exner pressure pi.
        Examples of usage:
          - If pi_arr is provided, compute p from pi.
          - If P_arr is provided, compute p from P.
          (In reality, you might pass both or keep separate methods.)
        """
        P = args[0]
        pressure = args[
            1
        ]  # Boolean whether the pressure is given or the exner pressure
        if pressure:
            return P_to_pressure_numba(P, self.p_ref, self.cp, self.cv)
        else:
            return exner_to_pressure_numba(P, self.p_ref, self.cp, self.R)

    def sound_speed(self, *args, **kwargs):
        """
        Calculate sound speed given density and pressure.
        Speed of sound can be approximated via gamma = cp/cv, or a more
        specialized formula. Here we use c = sqrt(gamma * p / rho).

        Parameters
        ----------
        rho : np.ndarray
            density
        p : np.ndarray
            The real pressure

        Returns
        -------
        np.ndarray
            the speed of sound as an array
        """
        rho = args[0]
        p = args[1]
        gamma = self.cp / self.cv
        return exner_sound_speed_numba(rho, p, gamma)
