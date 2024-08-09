import numpy as np
import numpy.testing as npt


class NodeGrid:
    """Initialize a general grid with x, y, and z

    Attributes
    ----------
    ranges : 2D np.ndarray of shape (3, 2)
        ranges of coordinates in each direction
    ninodes : np.ndarray of shape (3,)
        array of number of inner nodes in each direction (inner nodes are nodes that are not ghost nodes)
    nghosts : np.ndarray of shape (3,)
        array of number of ghosts cells in each direction
    nnodes : np.ndarray of shape (3,)
        array of inner nodes in each direction
    dim : int
        dimension of the grid
    L : np.ndarray or list of shape (3, 1)
        length of intervals in each coordinate direction
    nnx, nny, nnz : int
        number of nodes in each direction
    ngx, ngy, ngz : int
        number of ghost cells in each direction
    nix, niy, niz : int
        number of inner nodes in each direction
    xlims, ylims, zlims : ndarray of shape (2,)
        upper and lower bounds of the intervals in each direction
    dx, dy, dz : float
        the fineness of discretization in each direction
    ds : np.ndarray
        array of discretization fineness in each direction
    outer_xlims, outer_xlims, outer_xlims : ndarray of shape (2, )
        array of limits in each direction by considering the ghost nodes
    x, y, z : ndarray of shape (nnodes, )
        array of coordinates in each direction by considering the ghost cells
    ix, iy, iz : ndarray of shape (inodes, )
        array of coordinates of inner nodes in each direction

    """

    def __init__(
        self, ranges=np.zeros((3, 2)), ninodes=np.zeros(3), nghosts=np.zeros(3)
    ):
        """Initialize a general grid with x, y, and z

        Parameters
        ----------
        ranges : 2D ndarray or list of shape (3, 2)
            ranges of coordinates in each direction
        ninodes : ndarray or list of shape (3,)
            number of the inner nodes in each direction = limits + number of nodes in between
        nghosts : ndarray or list of shape (3,)
            number of ghosts cells in each direction

        """
        self.L = ranges[:, 1] - ranges[:, 0]
        self.dim = np.count_nonzero(ninodes)

        self.ranges = np.array(ranges)
        self.nghosts = np.array(nghosts)
        self.ninodes = np.array(ninodes)
        self.nnodes = np.zeros(3, dtype=int)
        self.nnodes = self.ninodes + 2 * self.nghosts

        assert np.all(self.L >= 0)

        if self.L[0] == 0:
            raise AssertionError("The problem should have at least one dimension")

        # Check for a given active dimension, the given number of input nodes is at least 2
        if np.any(self.ninodes[self.L > 0] < 2):
            raise AssertionError(
                " The input nodes given for an active dimension must be at least 2"
            )

        assert np.all(self.nnodes >= 0) and np.all(self.nghosts >= 0)

        if self.dim != np.count_nonzero(self.L) or self.dim < np.count_nonzero(
            self.nghosts
        ):
            raise AssertionError(
                "The number of dimensions in coordinates and nodes does not match"
            )

        self.nix, self.niy, self.niz = self.ninodes
        self.nnx, self.nny, self.nnz = self.nnodes
        self.ngx, self.ngy, self.ngz = self.nghosts

        self.xlims, self.ylims, self.zlims = self.ranges

        # -1 since the number of intervals
        # (and therefore ds) is 1 less than the number of nodes
        self.dx = self.L[0] / (self.ninodes[0] - 1) if self.ninodes[0] >= 2 else 0
        self.dy = self.L[1] / (self.ninodes[1] - 1) if self.ninodes[1] >= 2 else 0
        self.dz = self.L[2] / (self.ninodes[2] - 1) if self.ninodes[2] >= 2 else 0
        self.ds = np.array([self.dx, self.dy, self.dz])

        self.outer_xlims = self._compute_outer_lims(self.xlims, self.ngx, self.ds[0])
        self.outer_ylims = self._compute_outer_lims(self.ylims, self.ngy, self.ds[1])
        self.outer_zlims = self._compute_outer_lims(self.zlims, self.ngz, self.ds[2])

        self.x = self._coordinates(self.outer_xlims, self.nnx)
        self.y = self._coordinates(self.outer_ylims, self.nny) if self.nny > 1 else 0
        self.z = self._coordinates(self.outer_zlims, self.nnz) if self.nnz > 1 else 0

        self.ix = self._coordinates(self.xlims, self.nix)
        self.iy = self._coordinates(self.ylims, self.niy) if self.niy > 1 else 0
        self.iz = self._coordinates(self.ylims, self.niy) if self.niz > 1 else 0

        zeros = np.array([0, 0])
        if self.dim == 1:
            npt.assert_array_equal(self.ylims, zeros)
            npt.assert_array_equal(self.zlims, zeros)
            assert self.niy == 0 and self.niz == 0
            assert self.ngy == 0 and self.ngz == 0
            assert self.L[0] > 0
        if self.dim == 2:
            npt.assert_array_equal(self.zlims, zeros)
            assert self.niz == 0
            assert self.ngz == 0
            assert self.L[0] > 0 and self.L[1] > 0
        if self.dim == 3:
            assert self.L[0] > 0 and self.L[1] > 0 and self.L[2] > 0

    @staticmethod
    def _compute_outer_lims(lim, ng, ds):
        """Computes the outer lims of the grid in each direction"""
        return np.array([lim[0] - ng * ds, lim[1] + ng * ds])

    @staticmethod
    def _coordinates(lims, n):
        """Compute coodinates of the outer NODES"""
        return np.linspace(lims[0], lims[1], n)


class CellGrid(NodeGrid):
    """Class of cell grid. The unmentioned attributes of the class are overriden attributes of the parent class."""

    def __init__(
        self, ranges=np.zeros((3, 2)), ninodes=np.zeros(3), nghosts=np.zeros(3)
    ):
        super().__init__(ranges, ninodes, nghosts)
        new_begin = self.ranges[:, 0] - self.nghosts * self.ds
        new_end = self.ranges[:, 1] + self.nghosts * self.ds
        self.xlims, self.ylims, self.zlims = np.column_stack((new_begin, new_end))

        self.ninodes = self.ninodes - 1
        self.nnodes = self.nnodes - 1
        self.nix, self.niy, self.niz = self.ninodes
        self.nnx, self.nny, self.nnz = self.nnodes

        # self.x = (self.x + self.ds[0] / 2)[:-1]
        # self.y = (self.y + self.ds[1] / 2)[:-1]
        # self.z = (self.z + self.ds[2] / 2)[:-1]

        self.xlims, self.ylims, self.zlims = np.column_stack(
            (self.ranges[:, 0] + self.ds / 2, self.ranges[:, 1] - self.ds / 2)
        )

        self.outer_xlims = self._compute_outer_lims(self.xlims, self.ngx, self.ds[0])
        self.outer_ylims = self._compute_outer_lims(self.ylims, self.ngy, self.ds[1])
        self.outer_zlims = self._compute_outer_lims(self.zlims, self.ngz, self.ds[2])

        self.x = self._coordinates(self.outer_xlims, self.nnx)
        self.y = self._coordinates(self.outer_ylims, self.nny) if self.nny > 0 else 0
        self.z = self._coordinates(self.outer_zlims, self.nnz) if self.nnz > 0 else 0

        self.ix = self._coordinates(self.xlims, self.nix)
        self.iy = self._coordinates(self.ylims, self.niy) if self.niy > 0 else 0
        self.iz = self._coordinates(self.ylims, self.niy) if self.niz > 0 else 0


def main():
    grid = NodeGrid(np.array([[0, 1], [0, 2], [0, 3]]), [3, 3, 2], [1, 2, 0])
    cell = CellGrid(np.array([[0, 1], [0, 2], [0, 3]]), [3, 3, 2], [1, 2, 0])
    print(cell.x)
    print(node.x)


if __name__ == "__main__":
    main()
