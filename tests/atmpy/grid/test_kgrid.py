import numpy as np
import pytest
from atmpy.grid.kgrid import Grid  # Assuming the Grid class is in a file named grid.py


class TestKgrid:

    def test_empty_grid(self):
        with pytest.raises(ValueError):
            Grid(nx=10, x_start=0, x_end=0, ngx=None)
        with pytest.raises(ValueError):
            Grid(nx=10, x_start=0, x_end=0, ny=10, y_start=0, y_end=0, ngy=0)
        with pytest.raises(ValueError):
            Grid(
                nx=10,
                x_start=0,
                x_end=0,
                ny=1,
                y_start=0,
                y_end=0,
                ngy=2,
                nz=10,
                z_start=0,
                z_end=0,
            )

    def test_negative_number_of_cells(self):
        with pytest.raises(ValueError):
            Grid(nx=-1, x_start=0, x_end=0)
        with pytest.raises(ValueError):
            Grid(nx=10, x_start=0, x_end=0, ny=-1, y_start=0, y_end=1, ngy=1)
        with pytest.raises(ValueError):
            Grid(
                nx=10,
                x_start=0,
                x_end=0,
                ny=1,
                y_start=0,
                y_end=0,
                ngy=1,
                nz=-1,
                z_start=0,
                z_end=0,
                ngz=1,
            )

    def test_grid_initialization_1d(self):
        grid = Grid(nx=10, x_start=0.0, x_end=1.0, ngx=2)
        assert grid.ndim == 1
        assert grid.nx == 10
        assert grid.ngx == 2
        assert grid.ncx_total == 14  # nx + 2 * ngx
        assert len(grid.x_cells) == grid.ncx_total
        assert len(grid.x_nodes) == grid.ncx_total + 1

    def test_grid_initialization_2d(self):
        grid = Grid(
            nx=10, x_start=0.0, x_end=1.0, ny=20, y_start=0.0, y_end=2.0, ngx=2, ngy=3
        )
        assert grid.ndim == 2
        assert grid.nx == 10
        assert grid.ny == 20
        assert grid.ngx == 2
        assert grid.ngy == 3
        assert grid.ncx_total == 14  # nx + 2 * ngx
        assert grid.ncy_total == 26  # ny + 2 * ngy
        assert len(grid.x_cells) == grid.ncx_total
        assert len(grid.y_cells) == grid.ncy_total
        assert len(grid.x_nodes) == grid.ncx_total + 1
        assert len(grid.y_nodes) == grid.ncy_total + 1

    def test_grid_initialization_3d(self):
        grid = Grid(
            nx=10,
            x_start=0.0,
            x_end=1.0,
            ny=20,
            y_start=0.0,
            y_end=2.0,
            nz=30,
            z_start=0.0,
            z_end=3.0,
            ngx=2,
            ngy=3,
            ngz=4,
        )
        assert grid.ndim == 3
        assert grid.nx == 10
        assert grid.ny == 20
        assert grid.nz == 30
        assert grid.ngx == 2
        assert grid.ngy == 3
        assert grid.ngz == 4
        assert grid.ncx_total == 14  # nx + 2 * ngx
        assert grid.ncy_total == 26  # ny + 2 * ngy
        assert grid.ncz_total == 38  # nz + 2 * ngz
        assert len(grid.x_cells) == grid.ncx_total
        assert len(grid.y_cells) == grid.ncy_total
        assert len(grid.z_cells) == grid.ncz_total
        assert len(grid.x_nodes) == grid.ncx_total + 1
        assert len(grid.y_nodes) == grid.ncy_total + 1
        assert len(grid.z_nodes) == grid.ncz_total + 1

    def test_cell_mesh_1d(self):
        # 1D grid
        grid = Grid(nx=10, x_start=0.0, x_end=1.0, ngx=2)
        # Check that cell_centers is defined and has correct size
        assert hasattr(grid, "cell_mesh"), "Grid should have 'cell_mesh' attribute"
        assert grid.cell_mesh[0].shape == (grid.ncx_total,), "cell_mesh shape mismatch"
        # Check if cell centers are spaced by dx
        dx = (grid.x_end - grid.x_start) / grid.nx
        np.testing.assert_allclose(
            np.diff(grid.cell_mesh[grid.ngx : -grid.ngx]),
            dx,
            err_msg="Cell centers should be evenly spaced",
        )

    def test_cell_mesh_2d(self):
        # 2D grid
        grid = Grid(
            nx=10, x_start=0.0, x_end=1.0, ngx=2, ny=15, y_start=0.0, y_end=1.5, ngy=2
        )
        assert hasattr(grid, "cell_mesh"), "Grid should have 'cell_mesh' attribute"
        Xc, Yc = grid.cell_mesh
        assert Xc.shape == (grid.ncx_total, grid.ncy_total), "Xc shape mismatch"
        assert Yc.shape == (grid.ncx_total, grid.ncy_total), "Yc shape mismatch"

    def test_cell_mesh_3d(self):
        pass

    def test_node_mesh_1d(self):
        # 1D grid
        grid = Grid(nx=10, x_start=0.0, x_end=1.0, ngx=2)
        assert hasattr(grid, "node_mesh"), "Grid should have 'node_mesh' attribute"
        assert grid.node_mesh[0].shape == (grid.ncx_total + 1,), "nodes shape mismatch"
        # Check node spacing
        dx = (grid.x_end - grid.x_start) / grid.nx
        np.testing.assert_allclose(
            np.diff(grid.node_mesh[grid.ngx : -grid.ngx]),
            dx,
            err_msg="Nodes should be evenly spaced",
        )

    def test_node_mesh_2d(self):
        # 2D grid
        grid = Grid(
            nx=5, x_start=0.0, x_end=1.0, ngx=1, ny=5, y_start=0.0, y_end=1.0, ngy=1
        )
        assert hasattr(grid, "node_mesh"), "Grid should have 'node_mesh' attribute"
        Xn, Yn = grid.node_mesh
        assert Xn.shape == (grid.ncx_total + 1, grid.ncy_total + 1)
        assert Yn.shape == (grid.ncx_total + 1, grid.ncy_total + 1)

    def test_node_mesh_3d(self):
        pass

    def test_get_inner_cells_1d(self):
        grid = Grid(nx=10, x_start=0.0, x_end=1.0, ngx=2)
        inner_cells_slice = grid.get_inner_cells()
        var_cells = np.arange(grid.ncx_total)
        inner_cells = var_cells[inner_cells_slice]
        assert len(inner_cells) == grid.nx

    def test_get_inner_cells_2d(self):
        grid = Grid(
            nx=10, x_start=0.0, x_end=1.0, ny=20, y_start=0.0, y_end=2.0, ngx=2, ngy=3
        )
        inner_cells_slice = grid.get_inner_cells()
        var_cells = np.zeros((grid.ncx_total, grid.ncy_total))
        inner_cells = var_cells[inner_cells_slice]
        assert inner_cells.shape == (grid.nx, grid.ny)

    def test_get_inner_cells_3d(self):
        grid = Grid(
            nx=5,
            x_start=0.0,
            x_end=1.0,
            ny=5,
            y_start=0.0,
            y_end=1.0,
            nz=5,
            z_start=0.0,
            z_end=1.0,
            ngx=1,
            ngy=1,
            ngz=1,
        )
        inner_cells_slice = grid.get_inner_cells()
        var_cells = np.zeros((grid.ncx_total, grid.ncy_total, grid.ncz_total))
        inner_cells = var_cells[inner_cells_slice]
        assert inner_cells.shape == (grid.nx, grid.ny, grid.nz)

    def test_get_inner_cells_invalid_dimension(self):
        grid = Grid(nx=10, x_start=0.0, x_end=1.0)
        grid.ndim = 4  # Invalid dimension
        with pytest.raises(ValueError):
            grid.get_inner_cells()

    def test_apply_boundary_conditions_cells_1d(self):
        pass

    def test_apply_boundary_conditions_nodes_1d(self):
        pass

    def test_invalid_number_of_ghost_cells(self):
        with pytest.raises(ValueError):
            Grid(nx=10, x_start=0.0, x_end=1.0, ngx=0)

    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            # Missing y parameters for 2D grid
            Grid(nx=10, x_start=0.0, x_end=1.0, ny=10)
        with pytest.raises(ValueError):
            # Negative number of ghost cells
            Grid(nx=10, x_start=0.0, x_end=1.0, ngx=-1)

    def test_boundary_cells_slices_1d(self):
        grid = Grid(nx=10, x_start=0.0, x_end=1.0, ngx=2)
        left, right = grid.get_boundary_cells()
        var_cells = np.arange(grid.ncx_total)
        left_boundary = var_cells[left]
        right_boundary = var_cells[right]
        assert len(left_boundary) == grid.ngx
        assert len(right_boundary) == grid.ngx

    def test_boundary_nodes_slices_1d(self):
        grid = Grid(nx=10, x_start=0.0, x_end=1.0, ngx=2)
        left, right = grid.get_boundary_nodes()
        var_nodes = np.arange(grid.ncx_total + 1)
        left_boundary = var_nodes[left]
        right_boundary = var_nodes[right]
        assert len(left_boundary) == grid.ngx
        assert len(right_boundary) == grid.ngx - 1

    def test_apply_boundary_conditions_cells_2d(self):
        pass

    def test_apply_boundary_conditions_nodes_2d(self):
        pass

    def test_apply_boundary_conditions_cells_3d(self):
        pass

    def test_apply_boundary_conditions_nodes_3d(self):
        pass

    def test_inner_nodes_slices_1d(self):
        grid = Grid(nx=10, x_start=0.0, x_end=1.0, ngx=2)
        inner_nodes_slice = grid.get_inner_nodes()
        var_nodes = np.arange(grid.ncx_total + 1)
        inner_nodes = var_nodes[inner_nodes_slice]
        assert len(inner_nodes) == grid.nx + 1

    def test_inner_nodes_slices_2d(self):
        grid = Grid(
            nx=10, x_start=0.0, x_end=1.0, ny=20, y_start=0.0, y_end=2.0, ngx=2, ngy=2
        )
        inner_nodes_slice = grid.get_inner_nodes()
        var_nodes = np.zeros((grid.ncx_total + 1, grid.ncy_total + 1))
        inner_nodes = var_nodes[inner_nodes_slice]
        assert inner_nodes.shape == (grid.nx + 1, grid.ny + 1)

    def test_inner_nodes_slices_3d(self):
        grid = Grid(
            nx=5,
            x_start=0.0,
            x_end=1.0,
            ny=5,
            y_start=0.0,
            y_end=1.0,
            nz=5,
            z_start=0.0,
            z_end=1.0,
            ngx=1,
            ngy=1,
            ngz=1,
        )
        inner_nodes_slice = grid.get_inner_nodes()
        var_nodes = np.zeros(
            (grid.ncx_total + 1, grid.ncy_total + 1, grid.ncz_total + 1)
        )
        inner_nodes = var_nodes[inner_nodes_slice]
        assert inner_nodes.shape == (grid.nx + 1, grid.ny + 1, grid.nz + 1)

    def test_exceptions_in_methods(self):
        pass
