""" This class handles different definitions of discrete operators such as divergence and gradient. """

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple

class AbstractDiscreteOperators(ABC):
    """ Abstract class for discrete operators."""

    def __init__(self, ndim: int, dxyz: List[float]):
        self.ndim: int = ndim
        self.dxyz: List[float] = dxyz

    @abstractmethod
    def derivative(self, variable: np.ndarray, axis):
        """Calculates the nodal derivative of the given cell variable in the given direction.

        Parameters
        ----------
        variable : np.ndarray
            The cell defined variable on which the derivative is computed.
        axis : int
            The axis along which the derivative is computed.
        """
        pass

    @abstractmethod
    def gradient(self, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Calculate the discrete gradient of a given scalar field in 1D, 2D, or 3D. The variable 'p' is defined on
        the nodes.

        Parameters
        ----------
        p: np.ndarray
            The nodal variable."""
        pass

    @abstractmethod
    def divergence(self, vector: np.ndarray, *args, **kwargs):
        """ Calculate the divergence of the given variables. The variables are defined cell-centered. The result is
        on the nodes. The number of arguments passed should be the same as the number of dimensions.

        Parameters
        ----------
        vector : np.ndarray
            The vector consisting on which the divergence is calculated. The shape is (nx, [ny], [nz], num_components).

        Notes
        -----
        The order of the passed arguments matter. The divergence is calculated as follows: arg[0]_x + arg[1]_y + arg[2]_z
        """
        pass


class DiscreteOperators(AbstractDiscreteOperators):
    def __init__(self, ndim: int, dxyz: List[float]):
        super().__init__(ndim, dxyz)

    def derivative(self, variable: np.ndarray, axis: int):
        """Calculates the nodal derivative of the given cell variable in the given direction. This specific derivative
        is calculated by first finding values on the interfaces by averaging the cells and then use the interfaces around
        the node to calculate the derivative. See BK19 paper equation (31a, b).

        Parameters
        ----------
        variable : np.ndarray
            The cell-centered values.
        axis : int
            Axis along which to differentiate.

        Returns
        -------
          A numpy.ndarray containing the nodal derivative

        Notes
        -----
        The shape of the resulting array would be the same for derivative in any direction. The reason is simple:
        The np.diff reduces one shape in direction of derivation. The averaging then reduces one shape in the remaining directions.
        So the result would have shape (nx-1, [ny-1], [nz-1]).
        """
        # Discretization fineness
        ds = self.dxyz[axis]

        # Bring the differentiation axis to the front.
        u_flat = np.moveaxis(variable, axis, 0)

        # Compute the primary difference along the first axis.
        d = np.diff(u_flat, axis=0) / ds

        # Based on the dimensionality, average over the complementary axes.
        if self.ndim == 1:
            # 1D case: no additional averaging required.
            result = d
        elif self.ndim == 2:
            # 2D case: average along the second axis.
            # d has shape (n-1, m), so average adjacent values in m-direction.
            result = 0.5 * (d[:, :-1] + d[:, 1:])
        elif self.ndim == 3:
            # 3D case: average along both the second and third axes.
            # d has shape (n-1, m, p) and the nodal value is found by averaging
            # four neighboring interface values.
            result = (d[:, :-1, :-1] + d[:, :-1, 1:] + d[:, 1:, :-1] + d[:, 1:, 1:]) / 4.0
        else:
            raise ValueError("Only 1D, 2D, or 3D arrays are supported.")

        # Move the differentiation axis back to its original location.
        return np.moveaxis(result, 0, axis)

    def divergence(self, vector: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Calculates the divergence of the pressured-momenta vector (Pu, Pv, Pw). This works as the right hand side
            of the pressure equation in the euler steps.

            Parameters
            ----------
            vector : np.ndarray
                The momenta vector consisting of (u,v,w). The shape is (nx, [ny], [nz], num_components)


            Returns
            -------
            np.ndarray of shape (nx-1, [ny-1], [nz-1])
                The divergence of the velocity vector on the nodes.

            Notes
            -----
            This function calculates the divergence of the pressured-momenta vector on the nodes. The result can fill
            the inner nodes of a nodal variable as the result is of shape (nx-1, [ny-1], [nz-1]).
            """

        if vector.shape[-1] != self.ndim:
            raise ValueError("The number of arguments passed to the method must be the same as the number of dimensions.")

        if vector.shape[-1] > 3:
            raise ValueError("The number of arguments passed to the method must be the same as the number of dimensions.")

        U = np.zeros_like(args)
        axes = np.arange(vector.shape[-1])

        # Pre-allocation. The result should have one less element in all directions.
        Ux = np.zeros([nc-1 for nc in vector[..., 0].shape])

        for axis in range(vector.shape[-1]):
            Ux += self.derivative(vector[..., axis], axis)

        return Ux

    def gradient(self, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return p

if __name__ == "__main__":
    x = np.arange(30).reshape(5, 6)
    y = np.random.rand(5, 6)
    z = np.stack((x, y), axis=-1)

    ndim = 2
    dxyz = [0.1]*ndim

    print(x)
    print(y)
    print("-----------------------")
    obj = DiscreteOperators(ndim, dxyz)
    print(obj.derivative(x, 0))
    print(obj.derivative(y, 1))
    print("-------------------------")
    print(obj.divergence(z))
