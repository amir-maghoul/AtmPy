import numpy as np
from atmpy.grid.grid import Grid
import numpy.testing as npt
import pytest
from pytest_cases import PyTestCases1D


class TestGrid:

    TestCases = PyTestCases1D()
    success_1d = TestCases.success
    fail_1d = TestCases.fail

    @pytest.mark.parametrize("inputs", success_1d)
    def test_1d_success(self, inputs):
        grid1D = Grid(ranges=inputs[0], ninodes=inputs[1], nghosts=inputs[2])
        assert grid1D.dim == 1
        assert (
            grid1D.nny == 0 and grid1D.nnz == 0 and grid1D.ngy == 0 and grid1D.ngz == 0
        )
        npt.assert_array_equal(grid1D.nnodes, grid1D.ninodes + 2 * grid1D.nghosts)
        npt.assert_array_equal(grid1D.ranges[1:, :], 0)
        assert grid1D.dx
        assert grid1D.dy == 0 and grid1D.dz == 0

    @pytest.mark.parametrize("inputs", fail_1d)
    def test_1d_fail(self, inputs):
        with pytest.raises(AssertionError):
            Grid(ranges=inputs[0], ninodes=inputs[1], nghosts=inputs[2])

    success_2d = [(np.array([[0, 1], [0, 1], [0, 0]]), [3, 2, 0], [1, 2, 0])]
    fail_2d = [
        (np.array([[0, 1], [0, 1], [0, 0]]), [3, 0, 0], [3, 1, 0]),
        (np.array([[0, 1], [0, 0], [0, 1]]), [3, 2, 0], [3, 1, 0]),
        (np.array([[0, 1], [0, 1], [0, 0]]), [3, 1, 0], [3, 0, 1]),
        (np.array([[0, 1], [0, 1], [0, 0]]), [3, 0, 1], [3, 1, 0]),
        (np.array([[0, 1], [0, -1], [0, 0]]), [3, 1, 0], [3, 1, 0]),
    ]

    @pytest.mark.parametrize("inputs", fail_2d)
    def test_2d_fail(self, inputs):
        with pytest.raises(AssertionError):
            Grid(ranges=inputs[0], ninodes=inputs[1], nghosts=inputs[2])

    fail_3d = [
        (np.array([[0, 1], [0, 1], [0, 0]]), [3, 0, 0], [3, 1, 0]),
        (np.array([[0, 1], [0, 0], [0, 1]]), [3, 2, 1], [3, 1, 1]),
        (np.array([[0, 1], [0, 1], [0, 0]]), [3, 1, 0], [3, 0, 1]),
        (np.array([[0, 1], [0, 1], [0, 1]]), [3, 0, 1], [0, 0, 0]),
        (np.array([[0, 1], [0, -1], [0, 1]]), [3, 1, 0], [3, 1, 0]),
    ]

    @pytest.mark.parametrize("inputs", fail_3d)
    def test_3d_fail(self, inputs):
        with pytest.raises(AssertionError):
            Grid(ranges=inputs[0], ninodes=inputs[1], nghosts=inputs[2])


class TestCellGrid:
    def test_pass(self):
        assert True
