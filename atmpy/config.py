""" This module contains configuration values for the project. It contains the global and regime dependent constants
and  pysical quantities"""
from dataclasses import dataclass
from atmpy.data.TestCaseData import TestCaseData

@dataclass
class Config:
    """ Data class for the global values and quantities

    Attributes
    ----------
    alpha_g, alpha_w, alpha_p : int in {0, 1} (default = 1)
        values of the switches for different dynamical regimes
        alpha_g: geostrophic / non-geostrophic
        alpha_w: hydrostatic / non_hydrostatic
        alpha_p: compressible / soundproof

    """
    alpha_g: int = 1
    alpha_w: int = 1
    alpha_p: int = 1


class Thermodynamics:
    """ Data class for thermodynamic values and constants

    Parameters
    ----------
    data : atmpy.data.TestCaseData
        the data container of the test cases, it should contain needed values for calculation of the thermodynamic
        quantities

    Attributes
    ----------
    gamma : float
    gamma_inv : float
    Gamma : float
    Gamma_inv : float
    """
    def __init__(self, data : TestCaseData):
        self.gamma: float = data.gamma