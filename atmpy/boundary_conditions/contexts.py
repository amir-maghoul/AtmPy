"""This module holds contexts data classes for instantiation and application of boundary conditions"""

from dataclasses import dataclass, field
from typing import List, Tuple, TYPE_CHECKING
from atmpy.physics.thermodynamics import Thermodynamics

if TYPE_CHECKING:
    import numpy as np
    from atmpy.grid.kgrid import Grid
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
    gravity: Tuple[float] = (0.0, 1.0, 0.0)
    stratification: callable = lambda x: x
    thermodynamics: Thermodynamics = field(default_factory=Thermodynamics)
    is_compressible: bool = field(default=True)

@dataclass
class BoundaryConditionsConfiguration:
    """Data class of containing the context objects of all boundary conditions of all sides"""

    options: List[BCInstantiationOptions]
