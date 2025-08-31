import numpy as np
from typing import List

from atmpy.infrastructure.utility import direction_axis


def create_averaging_kernels(dimension: int) -> List[np.ndarray]:
    """
    Create a list of averaging kernels for 1D, 2D, or 3D,
    with one kernel per axis direction.

    For 1D (single direction):
        kernel = [0.5, 1.0, 0.5]

    For 2D:
        kernel_x = [[0.5, 1, 0.5], [0.5, 1, 0.5]]
        kernel_y = [[0.5, 1, 0.5], [0.5, 1, 0.5]].T

    For 3D:
        base_kernel = np.array([[1, 2, 1],
                                [2, 4, 2],
                                [1, 2, 1]], dtype=float)
        kernel_x = [base_kernel, base_kernel]
        kernel_y = [base_kernel, base_kernel]

    Parameters
    ----------
    dimension : int
        Either 1, 2, or 3.

    Returns
    -------
    List[np.ndarray]
        For 1D: [kx]
        For 2D: [kx, ky]
        For 3D: [kx, ky, kz]
    """
    if dimension == 1:
        kx = np.array([1.0, 1.0], dtype=np.float64)
        kx /= kx.sum()
        return [kx]

    elif dimension == 2:
        base_kernel = np.array([[0.5, 1.0, 0.5], [0.5, 1.0, 0.5]], dtype=np.float64)
        base_kernel /= base_kernel.sum()

        kx = base_kernel
        ky = base_kernel.T
        return [kx, ky]

    elif dimension == 3:
        base_2d = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float64)
        base_2d /= base_2d.sum()

        # kernel_x (shape: 2 x 3 x 3)
        kx = np.stack([base_2d, base_2d], axis=0)
        # kernel_y (shape: 3 x 2 x 3) - conceptually rotate to y-axis
        ky = np.stack([base_2d, base_2d], axis=1)
        # kernel_z (shape: 3 x 3 x 2) - conceptually rotate to z-axis
        kz = np.dstack([base_2d, base_2d])

        return [kx, ky, kz]

    else:
        raise ValueError("dimension must be 1, 2, or 3.")


def diffs_inner_slicing(ndim: int, direction: str):
    """Compute the correct indexing of the slopes container.

    Parameters
    ----------
    ndim : int
        The spatial dimension of the problem.
    direction_string : str
        The direction of the flow/in which the slopes are calculated.

    Returns
    -------
    Tuple
        consisting of
        left state indices
        right state indices
        the inner indices in the direction of the flow

    Notes
    -----
    In this function, the rights and inner slices happen (by circumstance) to be the same. But two different variables
    are chosen to duplicate to avoid usage/reading confusions.
    """

    lefts = [slice(None)] * (ndim + 1)
    rights = [slice(None)] * (ndim + 1)
    inner = [slice(None)] * (ndim + 1)

    if direction in ["x", "y", "z"]:
        direction = direction_axis(direction)
        lefts[direction] = slice(0, -2)
        rights[direction] = slice(1, -1)
        inner[direction] = slice(1, -1)
    else:
        raise ValueError("Invalid direction string")

    return (
        tuple(lefts),
        tuple(rights),
        tuple(inner),
    )
