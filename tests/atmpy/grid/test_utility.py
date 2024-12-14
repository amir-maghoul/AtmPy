import numpy as np
import pytest
from atmpy.grid.utility import *
from atmpy.grid.kgrid import Grid
from dataclasses import dataclass
from typing import List
from numpy.testing import assert_allclose


class TestUtility:

    def test_dimension_spec(self):
        dim = DimensionSpec(n=5, start=0, end=3, ng=2)
        assert dim.n == 5
        assert dim.start == 0
        assert dim.end == 3
        assert dim.ng == 2

    def test_to_grid_args_1d(self):
        dimensions = [DimensionSpec(5, 0, 3, 2)]
        args = to_grid_args(dimensions)
        expected = {"nx": 5, "x_start": 0, "x_end": 3, "ngx": 2}
        assert args == expected

    def test_to_grid_args_2d(self):
        dimensions = [DimensionSpec(5, 0, 3, 2), DimensionSpec(6, 1, 4, 3)]
        args = to_grid_args(dimensions)
        expected = {
            "nx": 5,
            "x_start": 0,
            "x_end": 3,
            "ngx": 2,
            "ny": 6,
            "y_start": 1,
            "y_end": 4,
            "ngy": 3,
        }
        assert args == expected

    def test_create_grid_1d(self):
        dimensions = [DimensionSpec(5, 0, 1, 2)]
        grid = create_grid(dimensions)
        assert grid.dimensions == 1
        assert grid.nx == 5
        assert grid.ngx == 2
        assert grid.x_start == 0
        assert grid.x_end == 1

    def test_create_grid_3d(self):
        dimensions = [
            DimensionSpec(5, 0, 1, 2),
            DimensionSpec(6, 1, 2, 2),
            DimensionSpec(7, 2, 3, 2),
        ]
        grid = create_grid(dimensions)
        assert grid.dimensions == 3
        assert grid.nx == 5 and grid.ngx == 2
        assert grid.ny == 6 and grid.ngy == 2
        assert grid.nz == 7 and grid.ngz == 2


    @pytest.fixture
    def grid_1d(self):
        dimensions = [DimensionSpec(n=5, start=0.0, end=5.0, ng=1)]
        return create_grid(dimensions)

    @pytest.fixture
    def grid_2d(self):
        dimensions = [
            DimensionSpec(n=4, start=0.0, end=4.0, ng=1),
            DimensionSpec(n=3, start=0.0, end=3.0, ng=2),
        ]
        return create_grid(dimensions)

    @pytest.fixture
    def grid_3d(self):
        dimensions = [
            DimensionSpec(n=2, start=0.0, end=2.0, ng=1),
            DimensionSpec(n=2, start=0.0, end=2.0, ng=1),
            DimensionSpec(n=2, start=0.0, end=2.0, ng=1),
        ]
        return create_grid(dimensions)

    def test_cell_to_node_average_1d(self, grid_1d):
        var_cells = np.array([0., 1., 2., 3., 4., 5., 6.])
        var_nodes = np.zeros(grid_1d.nshape)
        expected_nodes = np.array([0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 0])
        computed_nodes = cell_to_node_average(grid_1d, var_cells, var_nodes=var_nodes)
        assert_allclose(computed_nodes, expected_nodes, atol=1e-12)

    def test_node_to_cell_average_1d(self, grid_1d):
        var_nodes = np.array([0., 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.])
        var_cells = np.zeros(grid_1d.cshape)
        expected_cells = [0, 1, 2, 3, 4, 5, 0]
        computed_cells = node_to_cell_average(grid_1d, var_nodes, var_cells=var_cells)
        assert_allclose(computed_cells, expected_cells, atol=1e-12)

    def test_cell_to_node_average_2d(self, grid_2d):
        var_cells = np.arange(np.prod(grid_2d.cshape)).reshape(grid_2d.cshape)
        var_nodes = np.zeros(grid_2d.nshape)
        computed_nodes = cell_to_node_average(grid_2d, var_cells, var_nodes=var_nodes)
        recovered_cells = node_to_cell_average(grid_2d, computed_nodes, var_cells)
        assert_allclose(recovered_cells, var_cells, atol=1e-12)

    def test_node_to_cell_average_2d(self, grid_2d):
        var_cells = np.arange(np.prod(grid_2d.cshape)).reshape(grid_2d.cshape)
        var_nodes = cell_to_node_average(grid_2d, var_cells, np.zeros(grid_2d.nshape))
        computed_cells = node_to_cell_average(grid_2d, var_nodes, var_cells)
        assert_allclose(computed_cells, var_cells, atol=1e-12)

    def test_cell_to_node_average_3d(self, grid_3d):
        var_cells = np.arange(np.prod(grid_3d.cshape)).reshape(grid_3d.cshape)
        var_nodes = np.zeros(grid_3d.nshape)
        computed_nodes = cell_to_node_average(grid_3d, var_cells, var_nodes=var_nodes)
        recovered_cells = node_to_cell_average(grid_3d, computed_nodes, var_cells)
        assert_allclose(recovered_cells, var_cells, atol=1e-12)

    def test_node_to_cell_average_3d(self, grid_3d):
        var_cells = np.arange(np.prod(grid_3d.cshape)).reshape(grid_3d.cshape)
        var_nodes = cell_to_node_average(grid_3d, var_cells, np.zeros(grid_3d.nshape))
        computed_cells = node_to_cell_average(grid_3d, var_nodes, var_cells)
        assert_allclose(computed_cells, var_cells, atol=1e-12)
