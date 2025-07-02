"""This module encapsulates how extra conditions are defined for each BC class."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from atmpy.infrastructure.enums import (
    BoundarySide as BdrySide,
    BoundaryConditions as BdryType,
)


class ExtraBCOperation(ABC):
    """Abstract base class for an extra operation, potentially targeted."""

    def __init__(self, target_side: BdrySide, target_type: BdryType = None):
        """
        Initializes the operation with optional targeting.

        Parameters
        -----------
        target_side: BdrySide
            Apply only to the BC instance on this specific side. Takes precedence over target_type if both are set.
        target_type: BdryType
            Apply only to BC instances of this specific type (if target_side is None).
        """
        self.target_side: BdrySide = target_side
        self.target_type: BdryType = target_type

        if target_side:
            if not isinstance(target_side, BdrySide):
                raise TypeError("target_side must be of type 'BoundarySide' ")

        if target_type:
            if not isinstance(target_type, BdryType):
                raise TypeError("target_type must be of type 'BoundaryType' ")

    @abstractmethod
    def get_identifier(self) -> str:
        """Get the name of the operation."""
        pass

    def get_target_description(self) -> str:
        """Get the description of the operation."""
        if self.target_side:
            return f"side='{self.target_side}'"
        if self.target_type:
            return f"type={self.target_type.__name__}"
        return "all (broadcast)"


######################################
# Operation for Wall BCs
######################################
class WallAdjustment(ExtraBCOperation):
    def __init__(self, factor: float, **kwargs):  # Use kwargs for base init
        super().__init__(**kwargs)
        self.target_type = BdryType.WALL
        self.factor = factor

    def get_identifier(self) -> str:
        return f"WallAdjustment(factor={self.factor}, target={self.get_target_description()})"


class WallFluxCorrection(ExtraBCOperation):
    def __init__(self, factor: float, **kwargs):
        super().__init__(**kwargs)
        self.target_type = BdryType.WALL
        self.factor = factor

    def get_identifier(self) -> str:
        return f"WallFluxCorrection(factor={self.factor}, target={self.get_target_description()})"


######################################
# Operation for PERIODIC BCs
######################################
class PeriodicAdjustment(ExtraBCOperation):
    def __init__(self, factor: float, **kwargs):  # Use kwargs for base init
        super().__init__(**kwargs)
        self.target_type = BdryType.PERIODIC
        self.factor = factor

    def get_identifier(self) -> str:
        return f"PeriodicAdjustment(factor={self.factor}, target={self.get_target_description()})"


######################################
# Operation for Inlet BCs
######################################
class InletMassFlowCorrection(ExtraBCOperation):
    def __init__(self, target_mass_flow: float, relaxation: float, **kwargs):
        super().__init__(**kwargs)
        self.target_mass_flow = target_mass_flow
        self.relaxation = relaxation

    def get_identifier(self) -> str:
        return f"InletMassFlowCorrection(target={self.target_mass_flow}, relax={self.relaxation}, target={self.get_target_description()})"
