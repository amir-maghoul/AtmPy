import numpy as np


def create_averaging_kernels(dimension):
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
        return [kx]

    elif dimension == 2:
        kx = np.array([[0.5, 1.0, 0.5], [0.5, 1.0, 0.5]], dtype=float)
        ky = kx.T
        return [kx, ky]

    elif dimension == 3:
        base_2d = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=float)

        # kernel_x (shape: 2 x 3 x 3)
        kx = np.stack([base_2d, base_2d], axis=0)

        # kernel_y (shape: 3 x 2 x 3) - conceptually rotate to y-axis
        ky = np.stack([base_2d, base_2d], axis=1)

        # kernel_z (shape: 3 x 3 x 2) - conceptually rotate to z-axis
        kz = np.dstack([base_2d, base_2d])

        return [kx, ky, kz]

    else:
        raise ValueError("dimension must be 1, 2, or 3.")


if __name__ == "__main__":
    from scipy.ndimage import convolve

    # Suppose we want 2D kernels:
    kernels_2d = create_averaging_kernels(3)  # returns [kx, ky]
    kx, ky, kz = kernels_2d

    print(kx[0, ...])
    print(ky[:, 0, :])
    print(kz[..., 0])

    # Then convolve an image/array with any of these kernels:
    # convolved_x = convolve(image, kx, mode='constant', cval=0.0)
    # convolved_y = convolve(image, ky, mode='constant', cval=0.0)
