import numpy as np
from typing import List, Tuple


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
        kx = np.array([0.5, 1.0, 0.5], dtype=float)
        kx /= kx.sum()
        return [kx]

    elif dimension == 2:
        base_kernel = np.array([[0.5, 1.0, 0.5], [0.5, 1.0, 0.5]], dtype=float)
        base_kernel /= base_kernel.sum()

        kx = base_kernel
        ky = base_kernel.T
        return [kx, ky]

    elif dimension == 3:
        base_2d = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=float)
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


def directional_indices(
    ndim: int, direction_string: str, full: bool = True
) -> Tuple[Tuple[slice, ...], Tuple[slice, ...], Tuple[slice, ...], Tuple[slice, ...]]:
    """Compute the correct indexing of the flux vs. variable for the hll solvers and reconstruction.

    Parameters
    ----------
    ndim : int
        The spatial dimension of the problem.
    direction_string : str
        The direction of the flow/in which the slopes are calculated.
    full : bool, optional
        Whether to return the indices for full array or just a single variable within the array
        In this case, we need the slices for only one variable not the whole variable attribute
        Therefore we don't need the slices corresponding to the number of dimension (last entry of indices)

    Returns
    -------
    Tuple
        consisting of
        left state indices
        right state indices
        the inner indices in the direction of the flow
        the full inner indices of the whole array
    """
    # use ndim+1 to include the slices for the axis which corresponds to the number of variables in our cell_vars
    left_idx, right_idx, directional_inner_idx = (
        [slice(None)] * (ndim + 1),
        [slice(None)] * (ndim + 1),
        [slice(None)] * (ndim + 1),
    )
    if direction_string in ["x", "y", "z"]:
        direction = direction_mapping(direction_string)
        left_idx[direction] = slice(0, -1)
        right_idx[direction] = slice(1, None)
        directional_inner_idx[direction] = slice(1, -1)
    else:
        raise ValueError("Invalid direction string")
    inner_idx = [slice(1, -1)] * (ndim + 1)

    if full:
        return (
            tuple(left_idx),
            tuple(right_idx),
            tuple(directional_inner_idx),
            tuple(inner_idx),
        )
    else:
        return (
            tuple(left_idx)[:-1],
            tuple(right_idx)[:-1],
            tuple(directional_inner_idx)[:-1],
            tuple(inner_idx)[:-1],
        )


def direction_mapping(direction: str) -> int:
    """Utility function to map the string direction to its indices in variables."""
    direction_map = {"x": 0, "y": 1, "z": 2}
    return direction_map[direction]
