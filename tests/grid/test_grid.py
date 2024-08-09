import numpy as np
from atmpy.grid.grid import NodeGrid, CellGrid
import numpy.testing as npt
import pytest
from pytest_cases import PyTestCases1D, PyTestCases2D, PyTestCases3D


class TestGrid:

    ############################################################################
    #                                   1D
    ############################################################################

    TestCases = PyTestCases1D()
    success_1d = TestCases.success
    fail_1d = TestCases.fail

    @pytest.mark.parametrize("inputs", success_1d)
    def test_1d_success(self, inputs):
        nodegrid = NodeGrid(ranges=inputs[0], ninodes=inputs[1], nghosts=inputs[2])
        cellgrid = CellGrid(ranges=inputs[0], ninodes=inputs[1], nghosts=inputs[2])
        assert nodegrid.dim == 1
        assert (
            nodegrid.nny == 0 and nodegrid.nnz == 0 and nodegrid.ngy == 0 and nodegrid.ngz == 0
        )
        npt.assert_array_equal(nodegrid.nnodes, nodegrid.ninodes + 2 * nodegrid.nghosts)
        npt.assert_array_equal(nodegrid.ranges[1:, :], 0)
        assert nodegrid.dx
        assert nodegrid.dy == 0 and nodegrid.dz == 0
        npt.assert_array_equal(cellgrid.nnodes, nodegrid.nnodes - 1)
        npt.assert_array_equal(cellgrid.ninodes, nodegrid.ninodes - 1)
        assert len(cellgrid.x) == len(nodegrid.x) - 1
        assert cellgrid.y == nodegrid.y == 0
        assert cellgrid.z == nodegrid.z == 0

    @pytest.mark.parametrize("inputs", fail_1d)
    def test_1d_fail(self, inputs):
        with pytest.raises(AssertionError):
            NodeGrid(ranges=inputs[0], ninodes=inputs[1], nghosts=inputs[2])

    ############################################################################
    #                                   2D
    ############################################################################

    TestCases = PyTestCases2D()
    success_2d = TestCases.success
    fail_2d = TestCases.fail
    
    @pytest.mark.parametrize("inputs", success_2d)
    def test_2d_success(self, inputs):
        node = NodeGrid(ranges=inputs[0], ninodes=inputs[1], nghosts=inputs[2])
        cell = CellGrid(ranges=inputs[0], ninodes=inputs[1], nghosts=inputs[2])

        assert len(node.y) == len(cell.y) + 1
        assert cell.dim == node.dim == 2
        assert len(node.y) == len(cell.y) + 1

    @pytest.mark.parametrize("inputs", fail_2d)
    def test_2d_fail(self, inputs):
        with pytest.raises(AssertionError):
            NodeGrid(ranges=inputs[0], ninodes=inputs[1], nghosts=inputs[2])

    ############################################################################
    #                                   3D
    ############################################################################

    TestCases = PyTestCases3D()
    success_3d = TestCases.success
    fail_3d = TestCases.fail

    @pytest.mark.parametrize("inputs", success_3d)
    def test_3d_success(self, inputs):
        node = NodeGrid(ranges=inputs[0], ninodes=inputs[1], nghosts=inputs[2])
        cell = CellGrid(ranges=inputs[0], ninodes=inputs[1], nghosts=inputs[2])

        assert len(node.z) == len(cell.z) + 1
        assert cell.dim == node.dim == 3
        assert len(node.z) == len(cell.z) + 1

    @pytest.mark.parametrize("inputs", fail_3d)
    def test_3d_fail(self, inputs):
        with pytest.raises(AssertionError):
            NodeGrid(ranges=inputs[0], ninodes=inputs[1], nghosts=inputs[2])

