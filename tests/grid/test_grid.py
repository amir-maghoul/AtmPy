import numpy as np
from atmpy.grid.grid import Grid
import numpy.testing as npt
import pytest


class TestGrid:
    def test_init(self):
        grid = Grid()
        npt.assert_array_equal(grid.x, [0, 0])
        npt.assert_array_equal(grid.y, [0, 0])
        npt.assert_array_equal(grid.z, [0, 0])
        assert grid.nx == 0
        assert grid.ny == 0
        assert grid.nz == 0
        assert grid.ngx == 0
        assert grid.ngy == 0
        assert grid.ngz == 0

    success_1d = [(np.array([[0, 1], [0, 0], [0, 0]]), [3, 0, 0], [1, 0, 0])]
    fail_1d = [
        (np.zeros((3, 2)), [1, 0, 0], np.zeros(3)),
        (np.zeros((3, 2)), np.zeros(3), [1, 0, 0]),
        (np.array([[0, 1], [0, 0], [0, 0]]), np.zeros(3), np.zeros(3)),
        (np.array([[0, 1], [0, 0], [0, 0]]), [0, 1, 0], np.zeros(3)),
        (np.array([[0, 1], [0, 0], [0, 0]]), np.zeros(3), [0, 1, 0]),
        (np.array([[0, 1], [0, 0], [0, 0]]), [1, 0, 0], [0, 1, 0]),
        (np.array([[0, 1], [0, 0], [0, 0]]), [1, 0, 0], [0, 0, 1]),
        (np.array([[0, 1], [0, 0], [0, 0]]), [0, 1, 0], [0, 0, 1]),
        (np.array([[0, 1], [0, 0], [0, 0]]), [0, 1, 0], [0, 0, 1]),
        (np.array([[0, 0], [0, 0], [0, 1]]), [1, 0, 0], [1, 0, 0]),
        (np.array([[0, 0], [0, 1], [0, 0]]), [1, 0, 0], [1, 0, 0]),
        (np.array([[0, 0], [0, -1], [0, 0]]), [1, 0, 0], [1, 0, 0]),
        (np.array([[0, -1], [0, 0], [0, 0]]), [1, 0, 0], [1, 0, 0]),
    ]

    @pytest.mark.parametrize("inputs", success_1d)
    def test_1d_success(self, inputs):
        grid1D = Grid(ranges=inputs[0], nnodes=inputs[1], nghosts=inputs[2])
        assert grid1D.dim == 1
        assert grid1D.ny == 0 and grid1D.nz == 0 and grid1D.ngy == 0 and grid1D.ngz == 0
        npt.assert_array_equal(grid1D.ranges[1:, :], 0)

    @pytest.mark.parametrize("inputs", fail_1d)
    def test_1d_fail(self, inputs):
        with pytest.raises(AssertionError):
            grid1D = Grid(ranges=inputs[0], nnodes=inputs[1], nghosts=inputs[2])

    def test_pipeline_fail_test(self):
        with pytest.raises(AssertionError):
            grid1D = Grid(np.array([[0, 1], [0, 0], [0, 0]]), [3, 0, 0], [1, 0, 0])
