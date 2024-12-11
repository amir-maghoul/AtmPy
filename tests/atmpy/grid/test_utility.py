import numpy as np
import pytest
from atmpy.grid.utility import *
from atmpy.grid.kgrid import Grid
from dataclasses import dataclass
from typing import List


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

    def test_cell_to_node_average_1d(self):
        # Create a 1D grid
        dimensions = [DimensionSpec(10, 0.0, 1.0, 2)]
        grid = create_grid(dimensions)
        var_cells = np.arange(grid.nx_total, dtype=float)
        var_nodes = cell_to_node_average(grid, var_cells)
        # Check shape
        assert var_nodes.shape == (grid.nx_total + 1,)

        ngx = grid.ngx
        inner_slice = slice(ngx, -ngx)
        expected = 0.5 * (var_cells[ngx - 1 : -ngx] + var_cells[ngx : -ngx + 1])
        np.testing.assert_allclose(var_nodes[inner_slice], expected)

    def test_node_to_cell_average_1d(self):
        # Create a 1D grid
        dimensions = [DimensionSpec(10, 0.0, 1.0, 2)]
        grid = create_grid(dimensions)
        var_nodes = np.arange(grid.nx_total + 1, dtype=float)
        var_cells = node_to_cell_average(grid, var_nodes)
        # Check shape
        assert var_cells.shape == (grid.nx_total,)

        ngx = grid.ngx
        inner_slice = slice(ngx, -ngx)
        expected = 0.5 * (var_nodes[ngx:-ngx] + var_nodes[ngx + 1 : -ngx + 1])
        np.testing.assert_allclose(var_cells[inner_slice], expected)

    def test_cell_to_node_average_3d(self):
        dimensions = [
            DimensionSpec(4, 0.0, 1.0, 1),
            DimensionSpec(4, 0.0, 1.0, 1),
            DimensionSpec(4, 0.0, 1.0, 1),
        ]
        grid = create_grid(dimensions)
        nx_total, ny_total, nz_total = grid.nx_total, grid.ny_total, grid.nz_total
        var_cells = np.zeros((nx_total, ny_total, nz_total), dtype=float)

        # Fill var_cells with i+j+k pattern
        for i in range(nx_total):
            for j in range(ny_total):
                for k in range(nz_total):
                    var_cells[i, j, k] = i + j + k

        var_nodes = cell_to_node_average(grid, var_cells)
        assert var_nodes.shape == (nx_total + 1, ny_total + 1, nz_total + 1)

        # Check one inner node
        ngx, ngy, ngz = grid.ngx, grid.ngy, grid.ngz
        i, j, k = ngx, ngy, ngz
        # Compute expected manually
        cells = [
            (i - 1, j - 1, k - 1),
            (i, j - 1, k - 1),
            (i - 1, j, k - 1),
            (i, j, k - 1),
            (i - 1, j - 1, k),
            (i, j - 1, k),
            (i - 1, j, k),
            (i, j, k),
        ]
        expected_value = np.mean([var_cells[a, b, c] for (a, b, c) in cells])
        np.testing.assert_allclose(var_nodes[i, j, k], expected_value)

    def test_node_to_cell_average_3d(self):
        dimensions = [
            DimensionSpec(4, 0.0, 1.0, 1),
            DimensionSpec(4, 0.0, 1.0, 1),
            DimensionSpec(4, 0.0, 1.0, 1),
        ]
        grid = create_grid(dimensions)
        nx_total, ny_total, nz_total = grid.nx_total, grid.ny_total, grid.nz_total
        var_nodes = np.zeros((nx_total + 1, ny_total + 1, nz_total + 1), dtype=float)

        # Fill var_nodes with i+j+k
        for i in range(nx_total + 1):
            for j in range(ny_total + 1):
                for k in range(nz_total + 1):
                    var_nodes[i, j, k] = i + j + k

        # node_to_cell_average_3d is currently not implemented
        # We can check that it raises an error or skip this test.
        with pytest.raises(NotImplementedError):
            node_to_cell_average_3d(grid, var_nodes)

    def test_cell_to_node_average_2d_not_implemented(self):
        dimensions = [DimensionSpec(5, 0, 1, 2), DimensionSpec(5, 0, 1, 2)]
        grid = create_grid(dimensions)
        var_cells = np.zeros((grid.nx_total, grid.ny_total))
        with pytest.raises(NotImplementedError):
            cell_to_node_average_2d(grid, var_cells)

    def test_node_to_cell_average_2d_not_implemented(self):
        dimensions = [DimensionSpec(5, 0, 1, 2), DimensionSpec(5, 0, 1, 2)]
        grid = create_grid(dimensions)
        var_nodes = np.zeros((grid.nx_total + 1, grid.ny_total + 1))
        with pytest.raises(NotImplementedError):
            node_to_cell_average_2d(grid, var_nodes)
