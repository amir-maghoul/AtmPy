"""Utility module for the boundary handling"""
from curses import has_key
from typing import Dict, Any, Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from atmpy.grid.kgrid import Grid
from atmpy.physics.thermodynamics import Thermodynamics
from atmpy.infrastructure.enums import (
    BoundarySide as BdrySide,
    BoundaryConditions as BdryType
)


def side_direction_mapping(direction: str) -> Tuple[BdrySide, BdrySide]:
    """Returns the two sides of a given direction."""
    mapping = {
        "x": (BdrySide.LEFT, BdrySide.RIGHT),
        "y": (BdrySide.BOTTOM, BdrySide.TOP),
        "z": (BdrySide.FRONT, BdrySide.BACK),
    }
    return mapping[direction]

def create_params(bc_data:dict, side: BdrySide, condition: BdryType, **kwargs):
    """ Create or updates the parameters needed for creating boundary managers. It gets the bc_data dicts, adds the
    information of the new side to the dictionary.

    Parameters
    ----------
    bc_data: dict
        The boundary data for boundary manager object
    side: BdrySide
        The new side to be added as key to the bc_data
    condition: BdryType
        The type of boundary condition to be added to the bc_data

    Notes
    -----
    The key word arguments must contain the 'direction' information and the 'grid' information.

    Returns
    -------
    dict
        The updated bc_data

    """
    if kwargs.get("direction") is None:
        raise ValueError("The kwargs should contain 'direction' information.")
    direction: str = kwargs.get("direction")
    if kwargs.get("grid") is None:
        raise ValueError("The kwargs should contain 'grid' information.")
    grid: "Grid" = kwargs.get("grid")
    gravity = [0.0, 1.0, 0.0] if kwargs.get("gravity") is None else kwargs.get("gravity")
    stratification = kwargs.get("stratification")
    th: Thermodynamics = Thermodynamics() if kwargs.get("th") is None else kwargs.get("th")
    is_lamb: bool = False if kwargs.get("is_lamb") is None else kwargs.get("is_lamb")
    is_compressible: bool = False if kwargs.get("is_compressible") is None else kwargs.get("is_compressible")

    params = {
        "direction": direction,
        "grid": grid,
        "gravity": gravity,
        "stratification": stratification,
        "thermodynamics": th,
        "is_lamb": False,
        "is_compressible": True,
    }

    bc_data[side] = {"type": condition, "params": params}
    return bc_data



