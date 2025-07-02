""" Utility module for pressure solvers. """

from typing import Tuple, List


def laplacian_inner_slice(ng_all: List[Tuple[int, int]]):
    """Calculate the inner slice for the laplacian operator output.

    Parameters
    ----------
    ng_all: List[Tuple[int, int]]
        Tuple of number of ghost cells in each dimension and each side example: ((ngx, ngx),(ngy, ngy),(ngz, ngz))

    Notes
    -----
    The laplacian has two elements less in each direction than the input scalar field. Therefore, considering the
    number of ghost cells, we calculate the slice needed to get the inner points of the laplacian output.
    """

    return tuple(slice(ng - 1, -ng + 1) for (ng, _) in ng_all)


def single_element_slice(ndim: int, axis: int, element: int):
    """Calculate the slice for a single element in the given direction."""
    slices = [slice(None)]*ndim
    slices[axis] = element
    return tuple(slices)