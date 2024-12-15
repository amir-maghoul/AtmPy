from atmpy.grid.utility import *
from atmpy.grid.kgrid import Grid


class TestUtility:

    def test_dimensionspec_initiation(self):
        dimensions = DimensionSpec(n=3, start=0, end=1, ng=2)

        assert dimensions.n == 3
        assert dimensions.start == 0
        assert dimensions.end == 1
        assert dimensions.ng == 2

    def test_to_grid_args_1d(self):
        dimensions = [DimensionSpec(n=3, start=0, end=1, ng=2)]

        args = to_grid_args(dimensions)
        assert args["nx"] == 3
        assert args["x_start"] == 0
        assert args["x_end"] == 1
        assert args["ngx"] == 2

    def test_to_grid_args_2d(self):
        dimensions = [
            DimensionSpec(n=3, start=0, end=1, ng=2),
            DimensionSpec(n=4, start=1, end=2, ng=3),
        ]

        args = to_grid_args(dimensions)
        assert args["nx"] == 3
        assert args["x_start"] == 0
        assert args["x_end"] == 1
        assert args["ngx"] == 2
        assert args["ny"] == 4
        assert args["y_start"] == 1
        assert args["y_end"] == 2
        assert args["ngy"] == 3

    def test_to_grid_args_3d(self):

        dimensions = [
            DimensionSpec(n=3, start=0, end=1, ng=2),
            DimensionSpec(n=4, start=1, end=2, ng=3),
            DimensionSpec(n=5, start=2, end=3, ng=4),
        ]

        args = to_grid_args(dimensions)
        assert args["nx"] == 3
        assert args["x_start"] == 0
        assert args["x_end"] == 1
        assert args["ngx"] == 2
        assert args["ny"] == 4
        assert args["y_start"] == 1
        assert args["y_end"] == 2
        assert args["ngy"] == 3
        assert args["nz"] == 5
        assert args["z_start"] == 2
        assert args["z_end"] == 3
        assert args["ngz"] == 4

    def test_create_grid_1d(self):
        dimensions = [DimensionSpec(n=3, start=0, end=1, ng=2)]

        grid = create_grid(dimensions)
        assert grid.dimensions == 1
        assert grid.x_start == 0
        assert grid.x_end == 1
        assert grid.nx == 3
        assert grid.ngx == 2

    def test_create_grid_2d(self):
        dimensions = [
            DimensionSpec(n=3, start=0, end=1, ng=2),
            DimensionSpec(n=4, start=1, end=2, ng=3),
        ]
        grid = create_grid(dimensions)
        assert grid.dimensions == 2
        assert grid.x_start == 0
        assert grid.x_end == 1
        assert grid.nx == 3
        assert grid.ngx == 2
        assert grid.y_start == 1
        assert grid.y_end == 2
        assert grid.ny == 4
        assert grid.ngy == 3

    def test_create_grid_3d(self):
        dimensions = [
            DimensionSpec(n=3, start=0, end=1, ng=2),
            DimensionSpec(n=4, start=1, end=2, ng=3),
            DimensionSpec(n=5, start=2, end=3, ng=4),
        ]

        grid = create_grid(dimensions)
        assert grid.dimensions == 3
        assert grid.x_start == 0
        assert grid.x_end == 1
        assert grid.nx == 3
        assert grid.ngx == 2
        assert grid.y_start == 1
        assert grid.y_end == 2
        assert grid.ny == 4
        assert grid.ngy == 3
        assert grid.z_start == 2
        assert grid.z_end == 3
        assert grid.ngz == 4
        assert grid.nz == 5
