# --- Import functions to test ---
# Adjust the import below according to your project structure.
from atmpy.flux.reconstruction_utility import (
    calculate_variable_differences,
    calculate_slopes,
    calculate_amplitudes,
)
import numpy as np
from atmpy.infrastructure.utility import directional_indices, direction_axis


# Dummy PVI with attribute Y. In the production code PVI.Y is used as an index.
class DummyPVI:
    Y = 2


PVI = DummyPVI()


# --- Test for calculate_variable_differences ---
def test_calculate_variable_differences_1d():
    """
    Test that calculate_variable_differences returns the expected differences.
    For the test:
      - Let the primitives array have shape (4, 3) where the number of variables is 3.
      - PVI.Y is set to 2, so variable at index 0 uses np.diff along axis 0,
        variable at index 1 (the 'Y' variable) uses the reciprocal difference,
        and variable index 2 uses np.diff along axis 0.
    """
    # Create a small primitives array (4 points, 3 variables)
    primitives = np.random.rand(4, 3) * 10
    ndim = 1
    direction_str = "x"

    diffs = calculate_variable_differences(primitives, ndim, direction_str)
    expected_first = np.diff(primitives[..., :2], axis=0)
    expected_y = 1.0 / primitives[1:, 2] - 1.0 / primitives[:-1, 2]
    expected_rest = np.diff(primitives[..., 3:], axis=0)

    np.testing.assert_array_almost_equal(
        diffs[:-1, :2],
        expected_first,
        err_msg="Differences for variable 0 do not match expected values.",
    )
    # Check variable 1:
    np.testing.assert_array_almost_equal(
        diffs[:-1, 2],
        expected_y,
        err_msg="Differences for variable at PVI.Y do not match expected values.",
    )
    # Check variable 2:
    np.testing.assert_array_almost_equal(
        diffs[:-1, 3:],
        expected_rest,
        err_msg="Differences for variable after PVI.Y do not match expected values.",
    )
    # The last row should not have been updated and remain zeros.
    np.testing.assert_array_equal(
        diffs[-1], np.zeros(3), err_msg="The last row of differences should be zeros."
    )
    assert np.all(diffs[-1, :] == 0)


def test_calculate_variable_differences_2d():
    # Create a small 2D primitives array (4x5 grid, 3 variables)
    primitives = np.random.rand(4, 3, 5) * 10
    ndim = 2
    direction_str = "x"
    diffs = calculate_variable_differences(primitives, ndim, direction_str)

    expected_first = np.diff(primitives[..., :2], axis=0)
    expected_y = 1.0 / primitives[1:, :, 2] - 1.0 / primitives[:-1, :, 2]
    expected_rest = np.diff(primitives[..., 3:], axis=0)

    # Check variables 0 and 1:
    np.testing.assert_array_almost_equal(
        diffs[:-1, :, :2],
        expected_first,
        err_msg="Differences for variables 0 and 1 do not match expected values.",
    )

    # Check variable 2:
    np.testing.assert_array_almost_equal(
        diffs[:-1, :, 2],
        expected_y,
        err_msg="Differences for variable at PVI.Y do not match expected values.",
    )

    # Check variables after PVI.Y:
    np.testing.assert_array_almost_equal(
        diffs[:-1, :, 3:],
        expected_rest,
        err_msg="Differences for variables after PVI.Y do not match expected values.",
    )
    assert np.all(diffs[-1, :, :] == 0)

    direction_str = "y"
    diffs = calculate_variable_differences(primitives, ndim, direction_str)

    expected_first = np.diff(primitives[..., :2], axis=1)
    expected_y = 1.0 / primitives[:, 1:, 2] - 1.0 / primitives[:, :-1, 2]
    expected_rest = np.diff(primitives[..., 3:], axis=1)

    # Check variables 0 and 1:
    np.testing.assert_array_almost_equal(
        diffs[:, :-1, :2],
        expected_first,
        err_msg="Differences for variables 0 and 1 do not match expected values.",
    )

    # Check variable 2:
    np.testing.assert_array_almost_equal(
        diffs[:, :-1, 2],
        expected_y,
        err_msg="Differences for variable at PVI.Y do not match expected values.",
    )

    # Check variables after PVI.Y:
    np.testing.assert_array_almost_equal(
        diffs[:, :-1, 3:],
        expected_rest,
        err_msg="Differences for variables after PVI.Y do not match expected values.",
    )

    assert np.all(diffs[:, -1, :] == 0)


def test_calculate_variable_differences_3d():
    # Create a small 2D primitives array (4x5 grid, 3 variables)
    primitives = np.random.rand(4, 3, 6, 5) * 10
    ndim = 3
    direction_str = "z"
    diffs = calculate_variable_differences(primitives, ndim, direction_str)

    expected_first = np.diff(primitives[..., :2], axis=2)
    expected_y = 1.0 / primitives[:, :, 1:, 2] - 1.0 / primitives[:, :, :-1, 2]
    expected_rest = np.diff(primitives[..., 3:], axis=2)

    # Check variables 0 and 1:
    np.testing.assert_array_almost_equal(
        diffs[:, :, :-1, :2],
        expected_first,
        err_msg="Differences for variables 0 and 1 do not match expected values.",
    )

    # Check variable 2:
    np.testing.assert_array_almost_equal(
        diffs[:, :, :-1, 2],
        expected_y,
        err_msg="Differences for variable at PVI.Y do not match expected values.",
    )

    # Check variables after PVI.Y:
    np.testing.assert_array_almost_equal(
        diffs[:, :, :-1, 3:],
        expected_rest,
        err_msg="Differences for variables after PVI.Y do not match expected values.",
    )
    assert np.all(diffs[:, :, -1, :] == 0)


# --- Test for calculate_slopes ---
def dummy_limiter(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """
    A simple limiter function that returns the average of left and right slopes.
    """
    return 0.5 * (left + right)


def test_calculate_slopes_1d():
    # Create a sample diffs array with shape (4, 3)
    diffs = np.random.rand(4, 3) * 10
    ndim = 1
    direction_str = "x"

    limited_slopes = calculate_slopes(diffs, direction_str, dummy_limiter, ndim)
    left_variable_slopes = diffs[:3][:-1]  # rows 0 and 1 from diffs[0:3]
    right_variable_slopes = diffs[:3][1:]  # rows 1 and 2 from diffs[0:3]
    limited = dummy_limiter(left_variable_slopes, right_variable_slopes)

    # The limited slopes are placed into the inner slice (directional_inner_idx) of an array of zeros
    expected = np.zeros_like(diffs)
    expected[1:-1] = limited

    np.testing.assert_array_almost_equal(
        limited_slopes, expected, err_msg="Limited slopes do not match expected values."
    )


import numpy as np


def test_calculate_slopes_2d():
    # Create a sample diffs array with shape (4, 3, 3)
    diffs = np.random.rand(4, 3, 3) * 10
    ndim = 2
    direction_str = "x"  # Example direction for 2D

    limited_slopes = calculate_slopes(diffs, direction_str, dummy_limiter, ndim)

    # Assuming the limiter operates along both x and y directions
    left_variable_slopes = diffs[:-2, :, :]  # Exclude the last row and column
    right_variable_slopes = diffs[1:-1, :, :]  # Shifted by one in both directions
    limited = dummy_limiter(left_variable_slopes, right_variable_slopes)

    # The limited slopes are placed into the inner slice of an array of zeros
    expected = np.zeros_like(diffs)
    expected[1:-1, :, :] = limited

    np.testing.assert_array_almost_equal(
        limited_slopes,
        expected,
        err_msg="Limited slopes do not match expected values for 2D.",
    )

    direction_str = "y"
    limited_slopes = calculate_slopes(diffs, direction_str, dummy_limiter, ndim)

    # Assuming the limiter operates along both x and y directions
    left_variable_slopes = diffs[:, :-2, :]  # Exclude the last row and column
    right_variable_slopes = diffs[:, 1:-1, :]  # Shifted by one in both directions
    limited = dummy_limiter(left_variable_slopes, right_variable_slopes)

    # The limited slopes are placed into the inner slice of an array of zeros
    expected = np.zeros_like(diffs)
    expected[:, 1:-1, :] = limited

    np.testing.assert_array_almost_equal(
        limited_slopes,
        expected,
        err_msg="Limited slopes do not match expected values for 2D.",
    )


def test_calculate_slopes_3d():
    # Create a sample diffs array with shape (4, 3, 3, 3)
    diffs = np.random.rand(4, 3, 3, 3) * 10
    ndim = 3
    direction_str = "z"  # Example direction for 3D

    limited_slopes = calculate_slopes(diffs, direction_str, dummy_limiter, ndim)

    # Assuming the limiter operates along x, y, and z directions
    left_variable_slopes = diffs[
        :, :, :-2, :
    ]  # Exclude the last indices in each dimension
    right_variable_slopes = diffs[:, :, 1:-1, :]  # Shifted by one in all directions
    limited = dummy_limiter(left_variable_slopes, right_variable_slopes)

    # The limited slopes are placed into the inner slice of an array of zeros
    expected = np.zeros_like(diffs)
    expected[:, :, 1:-1, :] = limited

    np.testing.assert_array_almost_equal(
        limited_slopes,
        expected,
        err_msg="Limited slopes do not match expected values for 3D.",
    )


# --- Test for calculate_amplitudes ---
def test_calculate_amplitudes_1d():
    # Create sample slopes array of shape (4, 3)
    slopes = np.random.rand(4, 3) * 10
    # Create sample speed array of shape (4,)
    speed = np.array([0.5, 1.0, 1.5, 2.0])
    lmbda = 0.1

    # Test for left state (left=True, sign should be -1)
    amplitudes_left = calculate_amplitudes(slopes, speed, lmbda, True)
    expected_left = -0.5 * slopes * (1 - 0.1 * speed[..., np.newaxis])
    np.testing.assert_array_almost_equal(
        amplitudes_left,
        expected_left,
        err_msg="Left state amplitudes do not match expected values.",
    )

    # Test for right state (left=False, sign should be 1)
    amplitudes_right = calculate_amplitudes(slopes, speed, lmbda, False)
    expected_right = 0.5 * slopes * (1 + 0.1 * speed[..., np.newaxis])
    np.testing.assert_array_almost_equal(
        amplitudes_right,
        expected_right,
        err_msg="Right state amplitudes do not match expected values.",
    )


def test_calculate_amplitudes_2d():
    # Create sample slopes array of shape (4, 3, 2)
    slopes = np.random.rand(4, 3, 2) * 10
    # Create sample speed array of shape (4, 1)
    speed = np.arange(12).reshape(4, 3)
    lmbda = 0.1

    # Test for left state (left=True, sign should be -1)
    amplitudes_left = calculate_amplitudes(slopes, speed, lmbda, True)
    expected_left = -0.5 * slopes * (1 - 0.1 * speed[..., np.newaxis])
    np.testing.assert_array_almost_equal(
        amplitudes_left,
        expected_left,
        err_msg="Left state amplitudes (2D) do not match expected values.",
    )

    # Test for right state (left=False, sign should be 1)
    amplitudes_right = calculate_amplitudes(slopes, speed, lmbda, False)
    expected_right = 0.5 * slopes * (1 + 0.1 * speed[..., np.newaxis])
    np.testing.assert_array_almost_equal(
        amplitudes_right,
        expected_right,
        err_msg="Right state amplitudes (2D) do not match expected values.",
    )


def test_calculate_amplitudes_3d():
    import numpy as np

    # Create sample slopes array of shape (4, 3, 2, 5)
    slopes = np.random.rand(4, 3, 2, 5) * 10
    # Create sample speed array of shape (4, 1, 1)
    speed = np.arange(24).reshape(4, 3, 2)
    lmbda = 0.1

    # Test for left state (left=True, sign should be -1)
    amplitudes_left = calculate_amplitudes(slopes, speed, lmbda, True)
    expected_left = -0.5 * slopes * (1 - 0.1 * speed[..., np.newaxis])
    np.testing.assert_array_almost_equal(
        amplitudes_left,
        expected_left,
        err_msg="Left state amplitudes (3D) do not match expected values.",
    )

    # Test for right state (left=False, sign should be 1)
    amplitudes_right = calculate_amplitudes(slopes, speed, lmbda, False)
    expected_right = 0.5 * slopes * (1 + 0.1 * speed[..., np.newaxis])
    np.testing.assert_array_almost_equal(
        amplitudes_right,
        expected_right,
        err_msg="Right state amplitudes (3D) do not match expected values.",
    )
