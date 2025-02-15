import numpy as np
from atmpy.infrastructure.enums import BoundarySide


def boundary_map(ndim: int):
    """Create a boundary map for the name of the boundary sides for different dimensions.
        Notice the name of the sides are in upper case to be compatible with the boundary
        enums in py:mod:`atmpy.infrastructure.enums`.

    Parameters
    ----------
    ndim : int
        The dimension of the problem.

    Returns
    -------
    Dict
        dictionary of boundary sides with None values
    """

    directions = []
    if ndim == 1:
        directions = [BoundarySide.LEFT, BoundarySide.RIGHT]
    elif ndim == 2:
        directions = [
            BoundarySide.LEFT,
            BoundarySide.RIGHT,
            BoundarySide.TOP,
            BoundarySide.BOTTOM,
        ]
    elif ndim == 3:
        directions = [
            BoundarySide.LEFT,
            BoundarySide.RIGHT,
            BoundarySide.TOP,
            BoundarySide.BOTTOM,
            BoundarySide.FRONT,
            BoundarySide.BACK,
        ]
    else:
        raise ValueError("Unsupported dimensionality")

    return {direction: None for direction in directions}
