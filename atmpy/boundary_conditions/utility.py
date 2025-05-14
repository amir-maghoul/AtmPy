"""Utility module for the boundary handling"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from atmpy.grid.kgrid import Grid
from atmpy.physics.thermodynamics import Thermodynamics
from atmpy.infrastructure.enums import (
    BoundarySide as BdrySide,
    BoundaryConditions as BdryType,
)


def create_instatiation_context():
    pass


def create_application_context():
    pass


def create_operations_context(ndim: int):

    pass
