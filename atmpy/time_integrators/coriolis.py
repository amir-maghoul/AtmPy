"""This module is responsible for calculation on the coriolis operator"""

from typing import TYPE_CHECKING, Union, Any, Tuple
import numpy as np

if TYPE_CHECKING:
    from atmpy.physics.gravity import Gravity
    from atmpy.variables.variables import Variables
from atmpy.infrastructure.enums import VariableIndices as VI


class CoriolisOperator:
    """
    Encapsulates the handling of coriolis operator in the implicit time update. This is a part of operator splitting
    for stiff coriolis operator.
    """

    def __init__(
        self,
        coriolis_strength: Union[np.ndarray, list],
        gravity: "Gravity",
    ):
        self.strength: np.ndarray = np.array(coriolis_strength)
        self.gravity: "Gravity" = gravity

    def apply(
        self,
        u: np.ndarray,
        v: np.ndarray,
        w: np.ndarray,
        variables: "Variables",
        mpv: "MPV",
        direction: str,
        is_nonhydrostatic: bool,
        is_nongeostrophic: bool,
        Msq: float,
        dt: float,
    ) -> None:
        """Apply correction to momenta due to the coriolis effect. If there are no coriolis forces in any direction,
        do nothing.

        Parameters
        ----------
        u, v, w: np.ndarray
            momenta components to be updated
        variables : Variables
            The variable container containing the density and temperature variables.
        mpv : MPV
            The MPV object containing the Chi variable and its derivative method.
        dt : float
            The time step.
        """
        if self.strength is None or np.all(self.strength == 0):
            pass
        else:
            self._apply(variables, dt)

    def _apply(
        self,
        U: np.ndarray,
        V: np.ndarray,
        W: np.ndarray,
        variables: "Variables",
        mpv: "MPV",
        direction: str,
        is_nonhydrostatic: bool,
        is_nongeostrophic: bool,
        Msq: float,
        dt: float,
    ) -> None:

        # Prepare scalar values
        nonhydro = is_nonhydrostatic
        nongeo = is_nongeostrophic
        g: float = self.gravity.strength

        # Calculate dChi (the buoyancy in the momentum equation in the direction of the gravity)
        dChi = mpv.compute_dS_on_nodes()
        Theta = variables.cell_vars[..., VI.RHOY] / variables.cell_vars[..., VI.RHO]
        dChi_full_term = (dt**2) * (g / Msq) * dChi * Theta

        # Get elements of the inverse combined matrix (combination of switches, coriolis matrix and the buoyancy term)
        # See docstrings for more details.
        uu, uv, uw, vu, vv, vw, wu, wv, ww = self.calculate_inverse_coriolis(
            self.gravity.direction, dChi_full_term, nonhydro, nongeo, dt
        )

        # Matrix multiplication of the above matrix with the given momenta to be updated
        new_u = uu * U + uv * V + uw * W
        new_v = vu * U + vv * V + vw * W
        new_w = wu * U + wv * V + ww * W

        # Update the given variables.
        U[...] = new_u
        V[...] = new_v
        W[...] = new_w

    def calculate_inverse_coriolis(
        self,
        direction: str,
        dChi: np.ndarray,
        is_nonhydrostatic: bool,
        is_nongeostrophic: bool,
        dt: float,
    ):
        """Calculate the inverse the matrix appearing when solving the implicit euler to solve for momenta at the
        time step n+1. See BK16 eq. 24. We could write the matrix in the form (A + Omega + S). Since A and S are
        dependent of which direction the gravity is assumed, read the docstring of methods for the corresponding direction.
        """

        if direction == "y":
            return self._calculate_inverse_coriolis_y_direction(
                dChi, is_nonhydrostatic, is_nongeostrophic, dt
            )
        else:
            raise NotImplementedError(
                "The coriolis force for direction other that 'y' is not implemented yet."
            )

    def _calculate_inverse_coriolis_y_direction(
        self,
        dChi: np.ndarray,
        nonhydro: bool,
        nongeo: bool,
        dt: float,
    ):
        """Calculate the inverse the matrix appearing when solving the implicit euler to solve for momenta at the
        time step n+1. See BK16 eq. 24. We could write the matrix in the form (A + Omega + S)
        The components are as follows:
        - A = diag([a_g, a_w, a_g]): The 3x3 diagonal matrix of the switches. This is due to the LHS.
        - Omega: The usual Coriolis matrix
        - S: The singular matrix as a resulting effect of the buoyancy equation 24d. It only affects on the momentum
             in the direction of the gravity. For y-direction equal to diag([0, -dt*g*X_y*Theta, 0])

        Parameters
        ----------
        dChi : np.ndarray
            The term in the singular matrix. In y direction is equal to dt*g*X_y*Theta.

        Notes
        -----
        The direction of gravity in this method is set to be the second axis ('y' direction). That is why the hydrostatic
        switch a_w is on the second axis in A. And therefore S has nonzero value at the position (2,2).

        """

        # Calculate the timed coriolis strength
        coriolis = dt * self.strength

        determinant = self._calculate_determinant_y_direction(
            nonhydro, nongeo, coriolis, dChi, dt
        )

        #####################
        #### THE INVERSE ####

        uu, uv, uw = (
            self._inverse_matrix_first_row(nonhydro, nongeo, coriolis, dChi)
            / determinant
        )
        vu, vv, vw = (
            self._inverse_matrix_second_row(nonhydro, nongeo, coriolis, dChi)
            / determinant
        )
        wu, wv, ww = (
            self._inverse_matrix_third_row(nonhydro, nongeo, coriolis, dChi)
            / determinant
        )

        return uu, uv, uw, vu, vv, vw, wu, wv, ww

    def _inverse_matrix_first_row(
        self, nonhydro: bool, nongeo: bool, coriolis: np.ndarray, dChi: np.ndarray
    ) -> Tuple[Any, Any, Any]:
        """calculate the first row of the inverse matrix."""
        o1, o2, o3 = coriolis

        uu = nongeo * (nonhydro - dChi) + o1**2
        uv = nongeo * o3 + o1 * o2
        uw = o1 * o3 - o2 * (nonhydro - dChi)

        return uu, uv, uw

    def _inverse_matrix_second_row(
        self, nonhydro: bool, nongeo: bool, coriolis: np.ndarray, dChi: np.ndarray
    ) -> Tuple[Any, Any, Any]:
        """calculate the second row of the inverse matrix."""
        o1, o2, o3 = coriolis

        vu = o1 * o2 - nongeo * o3
        vv = nongeo**2 + o2**2
        vw = nongeo * o1 + o2 * o3

        return vu, vv, vw

    def _inverse_matrix_third_row(
        self, nonhydro: bool, nongeo: bool, coriolis: np.ndarray, dChi: np.ndarray
    ) -> Tuple[Any, Any, Any]:
        """Calculate the third row of the inverse matrix."""
        o1, o2, o3 = coriolis

        wu = o1 * o3 + o2 * (nonhydro - dChi)
        wv = o2 * o3 - nongeo * o1
        ww = nongeo * (nonhydro - dChi) + o3**2

        return wu, wv, ww

    def _calculate_determinant_y_direction(
        self,
        nonhydro: bool,
        nongeo: bool,
        coriolis: np.ndarray,
        dChi: np.ndarray,
        dt: float,
    ):
        """Calculate the determinant of the matrix involving the coriolis force. See calculate_inverse_coriolis_y_direction
        method for more details."""

        o1, o2, o3 = coriolis

        # Calculate the determinant
        determinant = (nonhydro - dChi) * (nongeo**2 + o2**2) + nongeo * (o1**2 + o3**2)
        return determinant
