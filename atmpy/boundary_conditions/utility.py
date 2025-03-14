"""Utility module for the boundary handling"""

import numpy as np


def get_grid_info_for_gravity():
    """Create a mapping between the given gravity axis as int and the grid information of that axis for the gravity
    handling.

    Parameters
    ----------
    gravity_axis : int
        The index of the gravity axis

    """
