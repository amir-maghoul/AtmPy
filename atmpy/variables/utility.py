import numpy as np
import scipy as sp


def cell_averaging(data: np.ndarray, kernel: np.ndarray):
    """
    Uses the scipy.ndimage.convolve to average the cell centered variable onto interface (Pu in flux) or onto nodes
    (dP/dpi in pressure solver).

    It performs a 'valid' convolution using the fast scipy.ndimage.convolve,
    then crops the result to match the 'valid' output shape.
    This is a drop-in replacement for fftconvolve(..., mode='valid').

    Parameters
    ----------
    data : np.ndarray
        The input N-dimensional array.
    kernel : np.ndarray
        The N-dimensional kernel.

    Returns
    -------
    np.ndarray
        The result of the convolution, with 'valid' shape.
    """
    if data.dtype != np.float64:
        data = np.array(data, dtype=np.float64)

    # Use 'reflect' mode for speed and good boundary handling; the cropped
    full_conv = sp.ndimage.convolve(data, kernel, mode="reflect")

    # Calculate the cropping required for each dimension.
    # This is the amount to remove from the start of the array.
    start_crop = (np.array(kernel.shape) - 1) // 2

    # This is the amount to remove from the end.
    end_crop_exclusive = np.array(data.shape) - (np.array(kernel.shape) // 2)

    # Build a slice object for N-dimensions
    slices = [slice(start, stop) for start, stop in zip(start_crop, end_crop_exclusive)]

    # Crop the full convolution to the 'valid' region
    return full_conv[tuple(slices)] / kernel.sum()


def cells_to_nodes_averaging(cell_data: np.ndarray) -> np.ndarray:
    """
    Averages cell-centered variable to the nodes at their corners.

    The result is an array with the shape of the inner nodes.

    Parameters
    ----------
    cell_data : np.ndarray
        The array of cell-centered data.

    Returns
    -------
    np.ndarray
        The node-centered data, with a shape corresponding to the inner nodes
        of the grid (e.g., (nx, ny) -> (nx-1, ny-1)).
    """
    ndim = cell_data.ndim
    if ndim > 3:  # Assuming the last axis is for variables
        ndim -= 1

    if ndim == 1:
        return 0.5 * (cell_data[:-1] + cell_data[1:])
    elif ndim == 2:
        return 0.25 * (
            cell_data[:-1, :-1]
            + cell_data[1:, :-1]
            + cell_data[:-1, 1:]
            + cell_data[1:, 1:]
        )
    elif ndim == 3:
        return 0.125 * (
            cell_data[:-1, :-1, :-1]
            + cell_data[1:, :-1, :-1]
            + cell_data[:-1, 1:, :-1]
            + cell_data[1:, 1:, :-1]
            + cell_data[:-1, :-1, 1:]
            + cell_data[1:, :-1, 1:]
            + cell_data[:-1, 1:, 1:]
            + cell_data[1:, 1:, 1:]
        )
    else:
        raise ValueError("Unsupported dimensionality for cell-to-node averaging.")
