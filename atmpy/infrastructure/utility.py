"""This module contains the utility functions that are shared in the whole project"""

from typing import Tuple

from atmpy.infrastructure.enums import BoundarySide, BoundarySide as BdrySide
from atmpy.infrastructure.enums import (
    VariableIndices as VI,
    PrimitiveVariableIndices as PVI,
)


def dimension_directions(ndim: int):
    """Returns the list of directions corresponding to the ndim."""
    if ndim == 1:
        directions = ["x"]
    elif ndim == 2:
        directions = ["x", "y"]
    elif ndim == 3:
        directions = ["x", "y", "z"]
    else:
        raise ValueError("Unsupported dimensionality")

    return directions


def directional_indices(
    ndim: int, direction_string: str, full: bool = True
) -> Tuple[Tuple[slice, ...], Tuple[slice, ...], Tuple[slice, ...]]:
    """Compute the correct indexing of the flux vs. variable for the hll solvers and reconstruction.

    Parameters
    ----------
    ndim : int
        The spatial dimension of the problem.
    direction_string : str
        The direction of the flow/in which the slopes are calculated.
    full : bool, optional
        Whether to return the indices for full array or just a single variable within the array
        In the latter case, we need the slices for only one variable not the whole variable attribute
        Therefore we don't need the slices corresponding to the number of dimension (last entry of indices)

    Returns
    -------
    Tuple
        consisting of
        left state indices
        right state indices
        the inner indices in the direction of the flow
    """
    # use ndim+1 to include the slices for the axis which corresponds to the number of variables in our cell_vars
    left_idx, right_idx, directional_inner_idx = (
        [slice(None)] * (ndim + 1),
        [slice(None)] * (ndim + 1),
        [slice(None)] * (ndim + 1),
    )
    if direction_string in ["x", "y", "z"]:
        direction = direction_axis(direction_string)
        left_idx[direction] = slice(0, -1)
        right_idx[direction] = slice(1, None)
        directional_inner_idx[direction] = slice(1, -1)
    else:
        raise ValueError("Invalid direction string")

    if full:
        return (
            tuple(left_idx),
            tuple(right_idx),
            tuple(directional_inner_idx),
        )
    else:
        return (
            tuple(left_idx)[:-1],
            tuple(right_idx)[:-1],
            tuple(directional_inner_idx)[:-1],
        )


def one_element_inner_slice(ndim: int, full: bool = True) -> Tuple[slice, ...]:
    inner_idx = [slice(1, -1)] * (ndim + 1)
    if full:
        return tuple(inner_idx)
    else:
        return tuple(inner_idx[:-1])


def direction_axis(direction: str) -> int:
    """Utility function to map the string direction to its indices in variables."""
    direction_map = {"x": 0, "y": 1, "z": 2}
    return direction_map[direction]


def boundary_map(ndim: int):
    """Create a boundary map for the name of the boundary sides for different ndim.
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


def zipped_direction(ndim: int):
    """Return the zipped string and integer of directions corresponding to the ndim."""
    if ndim == 1:
        dir_string = ["x"]
        dir_idx = [0]
    elif ndim == 2:
        dir_string = ["x", "y"]
        dir_idx = [0, 1]
    elif ndim == 3:
        dir_string = ["x", "y", "z"]
        dir_idx = [0, 1, 2]
    else:
        raise ValueError("Unsupported dimensionality")

    return zip(dir_string, dir_idx)


def side_direction_mapping(direction: str) -> Tuple[BdrySide, BdrySide]:
    """Returns the two sides of a given direction."""
    mapping = {
        "x": (BdrySide.LEFT, BdrySide.RIGHT),
        "y": (BdrySide.BOTTOM, BdrySide.TOP),
        "z": (BdrySide.FRONT, BdrySide.BACK),
    }
    return mapping[direction]


def momentum_index(axis: int):
    """Returns the velocity index for a given axis."""
    momenta = [VI.RHOU, VI.RHOV, VI.RHOW]
    return momenta[axis]


def velocity_index(axis: int):
    """Returns the primitive velocity index for a given axis."""
    velocities = [PVI.U, PVI.V, PVI.W]
    return velocities[axis]
