"""This module holds contexts data classes for instantiation and application of boundary conditions"""

from dataclasses import dataclass, field
from typing import List, Tuple, TYPE_CHECKING, Optional
from atmpy.physics.thermodynamics import Thermodynamics

if TYPE_CHECKING:
    from atmpy.grid.kgrid import Grid
    from atmpy.variables.multiple_pressure_variables import MPV
from atmpy.infrastructure.enums import (
    BoundarySide as BdrySide,
    BoundaryConditions as BdryType,
)


@dataclass
class BCApplicationContext:
    """Data class to contain the boundary condition application context."""

    is_nodal: bool = False


@dataclass
class BCInstantiationOptions:
    """Data class to contain the boundary condition instantiation parameters."""

    side: "BdrySide"
    type: "BdryType"
    direction: str
    grid: "Grid"
    mpv_boundary_type: Optional["BdryType"] = None
    gravity: Tuple[float] = (0.0, 10.0, 0.0)
    stratification: callable = lambda x: x
    thermodynamics: Thermodynamics = field(default_factory=Thermodynamics)
    is_compressible: bool = field(default=True)
    mpv: Optional["MPV"] = None

    def __post_init__(self):
        # Initialize the mpv_boundary_type to match the main boundary type in case there is no gravity.
        if self.type != BdryType.REFLECTIVE_GRAVITY and self.mpv_boundary_type is None:
            self.mpv_boundary_type = self.type


@dataclass
class BoundaryConditionsConfiguration:
    """Data class of containing the context objects of all boundary conditions of all sides"""

    options: List[BCInstantiationOptions]
