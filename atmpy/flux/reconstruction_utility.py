import numpy as np
from typing import Callable
from atmpy.flux.utility import directional_indices, direction_mapping
from atmpy.infrastructure.enums import PrimitiveVariableIndices as PVI


def calculate_variable_differences(
    primitives: np.ndarray,
    ndim: int,
    direction_str: str,
) -> np.ndarray:
    """Calculate the difference of primitive variables in the given direction and store it in an array with the same
    shape as the variable array. The differences are needed to be passed to the limiter function to limit the slope.

    Parameters
    ----------
    primitives : np.ndarray of shape (nx, [ny], [nz], num_vars)
        The array of primitive variables.
    ndim: int
        The spatial dimension of the variables.
    direction_str : str
        The direction of the flux calculation.

    Returns
    -------
    np.ndarray
        The array of differences of the primitive variables with one less element.

    Notes
    -----
    The ADVECTIVE value for scalar in the flux is not Theta anymore, it is Chi (X) so here in calculation of slopes
    and diffs, the PVI.Y variable is Chi not Theta. Therefore the difference is calculated by the inversed values of Theta.
    """

    direction = direction_mapping(direction_str)
    diffs = np.zeros_like(primitives)  # The final difference array]

    # Set the difference slice (one fewer element than the original array) in the corresponding direction
    left_idx, right_idx, _, _ = directional_indices(ndim, direction_str, full=False)

    # Apply np.diff in the direction which results in one less element
    # Notice the PVI.Y element is calculated differently and np.diff is not applied on it
    diffs[..., : PVI.Y][left_idx] = np.diff(primitives[..., : PVI.Y], axis=direction)
    diffs[..., PVI.Y][left_idx] = (
        1.0 / primitives[..., PVI.Y][right_idx] - 1.0 / primitives[..., PVI.Y][left_idx]
    )
    diffs[..., PVI.Y + 1 :][left_idx] = np.diff(
        primitives[..., PVI.Y + 1 :], axis=direction
    )
    return diffs


def calculate_slopes(
    diffs: np.ndarray,
    direction_str: str,
    limiter: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ndim: int,
) -> np.ndarray:
    """Calculate the slopes of the variables from their given difference array using the left and right values

    Parameters
    ----------
    diffs : np.ndarray of shape (nx, [ny], [nz], num_vars)
        Array of differences of the primitive variables. It should be calculated using the function
        :py:func:`atmpy.flux.reconstruction.calculate_differential_variables`
    direction_str : str
        Direction in which the slopes and the flow are calculated.
    limiter : Callable[[np.ndarray, np.ndarray], np.ndarray]
        The flux slope limiter passed as a function.
    ndim : int
        The spatial dimension of the variables.

    Returns
    -------
    np.ndarray of shape (nx, [ny], [nz], num_vars)
        The limited slopes of the primitive variables at interfaces

    Notes
    -----
    The limited slope has two fewer rows/columns in the direction of calculation than the original variables.
    """
    left_idx, right_idx, directional_inner_idx, _ = directional_indices(
        ndim, direction_str
    )
    # Use twice indexing: once to eliminate the extra zero due to the size difference between vars and differences
    # (differences should have one less element) and once to obtain the left values
    left_variable_slopes = diffs[left_idx][left_idx]
    right_variable_slopes = diffs[left_idx][right_idx]
    limited_slopes = np.zeros_like(diffs)
    # Since the limiter returns an array with 2 fewer rows/columns, it should be placed in the inner cells of
    # an array with the same shape as variables
    limited_slopes[directional_inner_idx] = limiter(
        left_variable_slopes, right_variable_slopes
    )
    return limited_slopes


def calculate_amplitudes(
    slopes: np.ndarray, speed: np.ndarray, lmbda: float, left: bool
) -> np.ndarray:
    """Compute left and right state amplitudes according to BK19 paper equation 22c and 22d

    Parameters
    ----------
    slopes : np.ndarray of shape (nx, [ny], [nz], num_vars)
        The array of slopes for all variables
    speed : np.ndarray of shape (nx, [ny], [nz])
        The flow speed at the interfaces in the given direction. This is calculated using the modified equation
        22e of BK19.
    lmbda : float
        The ratio of delta_t to delta_x
    left : bool
        Whether the left state should be calculated or the right state

    """

    sign = -1 if left else 1
    amplitudes = sign * (0.5 * slopes * (1 + sign * lmbda * speed[..., np.newaxis]))
    return amplitudes
