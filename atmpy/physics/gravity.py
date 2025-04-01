"""This module handles basics regarding the gravity. Fingind the axis, coordinates, momenta indices and etc."""

import numpy as np
from typing import Union, cast, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from atmpy.grid.kgrid import Grid
from atmpy.infrastructure.enums import (
    VariableIndices as VI,
    VariableIndices,
)  # assuming you have this enum


class Gravity:
    def __init__(self, gravity_vector: Union[np.ndarray, list], ndim: int):
        self.vector: np.ndarray = np.array(gravity_vector)
        self.ndim = ndim
        non_zero = np.nonzero(self.vector)[0]
        if len(non_zero) != 1:
            raise ValueError("Gravity vector must be nonzero in exactly one direction.")
        self.axis: int = cast(int, non_zero[0])
        if self.axis >= self.ndim:
            raise ValueError(
                f"""An {self.ndim}-dimensional problem cannot have gravity on axis {self.axis}. 
                The gravity should exist on the highest dimension."""
            )
        if self.axis == 0 and self.ndim != 1:
            raise ValueError(
                """In reflective gravity boundary condition for problems of more than one dimension, 
                the first axis is reserved for horizontal velocity. It cannot have gravity."""
            )
        self.strength: float = abs(self.vector[self.axis])

    @property
    def momentum_index(self) -> Tuple[int, int]:
        """Helper method to get the momentum variable index in the direction of gravity as the first output and the
        momentum in the nongravity direction as the second output. Notice since RHOU can never be the momentum in the
        gravity axis (or more clearly, first axis can never be the gravity axis), it is not included in the array
        """
        if self.axis == 1:
            return (VI.RHOV, VI.RHOW)
        elif self.axis == 2:
            return (VI.RHOW, VI.RHOV)
        else:
            raise ValueError(
                "Invalid gravity axis. Gravity cannot be applied on axis 0."
            )

    def get_coordinate_cells(self, grid: "Grid"):
        """
        Given a grid object, return the coordinate cells along the gravity axis.
        """
        return grid.get_cell_coordinates(self.axis)
