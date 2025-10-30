"""This module handles basics regarding the gravity. Fingind the axis, coordinates, momenta indices and etc.

The gravity axis throughout the project is assumed to be the second axis (index = 1).
Regardless of the dimension of the problem."""

import numpy as np
from typing import Union, cast, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from atmpy.grid.kgrid import Grid
from atmpy.infrastructure.enums import (
    VariableIndices as VI,
    VariableIndices,
)  # assuming you have this enum

#### Setting global constant for gravity axis ###
GRAVITY_AXIS: int = 1
GRAVITY_DIRECTION: str = "y"
GRAVITY_MOMENTUM_INDEX: int = VI.RHOV
PERPENDICULAR_MOMENTUM_INDEX: int = VI.RHOW


class Gravity:
    """The Gravity container. Create the needed tool to work with gravity throughout the project.

    Attributes
    ----------
    vector: np.ndarray
        The gravity vector.
    ndim: int
        The dimension of the problem
    axis: int
        The axis on which the gravity force exists.
    direction: str
        The axis on which the gravity force exists but in string. Values can be "x", "y", "z".
    strength: float
        The gravity strength
    momentum_index: Tuple[int, int]
        The index of the momentum in the direction of gravity and in the direction of non-gravity

    """

    def __init__(self, gravity_vector: Union[np.ndarray, list, tuple], ndim: int):
        self.vector: np.ndarray = np.array(gravity_vector)
        self.ndim = ndim
        self.axis = GRAVITY_AXIS
        self.direction: str = GRAVITY_DIRECTION
        self.strength = 0.0
        self.vertical_momentum_index = GRAVITY_MOMENTUM_INDEX
        self.perpendicular_momentum_index = PERPENDICULAR_MOMENTUM_INDEX

        non_zero = np.nonzero(self.vector)[0]
        if non_zero.size == 1:
            self.axis: int = cast(int, non_zero[0])
            if self.axis != GRAVITY_AXIS:
                raise ValueError(
                    f""" The axis {GRAVITY_AXIS} is the forced axis for gravity. """
                )
            self.strength: float = self.vector[self.axis]
        elif non_zero.size > 1:
            raise ValueError(
                "Gravity vector cannot have strength in more than one direction."
            )

    def get_coordinate_cells(self, grid: "Grid"):
        """
        Given a grid object, return the coordinate cells along the gravity axis.
        """
        return grid.get_cell_coordinates(self.axis)

    def horizontal_momentum_indices(self):
        """ Returns the indices of the horizontal momentum """
        momentum_indices = [VI.RHOU, VI.RHOV, VI.RHOW]
        horizontal_indices = [val for val in momentum_indices if val != self.vertical_momentum_index]
        return horizontal_indices
