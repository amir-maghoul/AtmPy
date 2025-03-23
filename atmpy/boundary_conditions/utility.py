"""Utility module for the boundary handling"""

from typing import Dict, Any, Tuple
from atmpy.infrastructure.enums import (
    BoundarySide as BdrySide,
)

def side_direction_mapping(direction: str) -> Tuple[BdrySide, BdrySide]:
    """ Returns the two sides of a given direction."""
    mapping = {
        "x": (BdrySide.LEFT, BdrySide.RIGHT),
        "y": (BdrySide.BOTTOM, BdrySide.TOP),
        "z": (BdrySide.FRONT, BdrySide.BACK),
    }
    return mapping[direction]
