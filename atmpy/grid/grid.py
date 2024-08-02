import numpy as np
import numpy.testing as npt


class Grid:
    def __init__(self, ranges=np.zeros((3, 2)), nnodes=np.zeros(3), nghosts=np.zeros(3)):
        """ Initialize a general grid with x, y, and z """
        self.dim = np.count_nonzero(nnodes)
        self.lims = ranges[:, 1] - ranges[:, 0]
        if self.dim != np.count_nonzero(self.lims) or self.dim != np.count_nonzero(nghosts):
            raise AssertionError("The number of dimensions in coordinates and nodes does not match")
        self.ranges = ranges
        self.nnodes = nnodes
        self.nghosts = nghosts
        zeros = np.array([0, 0])
        self.nx, self.ny, self.nz = self.nnodes[0], self.nnodes[1], self.nnodes[2]
        self.ngx, self.ngy, self.ngz = self.nghosts[0], self.nghosts[1], self.nghosts[2]
        self.x, self.y, self.z = self.ranges[0], self.ranges[1], self.ranges[2]

        assert np.all(self.lims >= 0)

        if self.dim == 1:
            npt.assert_array_equal(self.y, zeros)
            npt.assert_array_equal(self.z, zeros)
            assert self.ny == 0 and self.nz == 0
            assert self.ngy == 0 and self.ngz == 0
        if self.dim == 2:
            npt.assert_array_equal(self.z, zeros)
            assert self.nz == 0
            assert self.ngz == 0

def main():
    grid = Grid(np.array([[0, 0], [0, 0], [0, 1]]),  [1, 0, 0], [1, 0, 0])
    npt.assert_array_equal(grid.z, np.array([0, 0]))


if __name__ == "__main__":
    main()

