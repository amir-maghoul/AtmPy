"""This module handles solving the equation for the pressure variable including the Laplace/Poisson equation."""

import numpy as np
import scipy as sp
from typing import TYPE_CHECKING, Union, Tuple

from atmpy.infrastructure.enums import (
    VariableIndices as VI,
    BoundarySide as BdrySide,
    BoundaryConditions as BdryType,
)
from atmpy.boundary_conditions.bc_extra_operations import WallAdjustment
from atmpy.infrastructure.utility import one_element_inner_slice
from atmpy.physics.thermodynamics import Thermodynamics
from atmpy.pressure_solver.abstract_pressure_solver import AbstractPressureSolver
from atmpy.pressure_solver.utility import laplacian_inner_slice

if TYPE_CHECKING:
    from atmpy.pressure_solver.discrete_operations import AbstractDiscreteOperator
    from atmpy.pressure_solver.linear_solvers import ILinearSolver
    from atmpy.variables.variables import Variables
    from atmpy.variables.multiple_pressure_variables import MPV
    from atmpy.boundary_conditions.boundary_manager import BoundaryManager
    from atmpy.time_integrators.coriolis import CoriolisOperator
    from atmpy.physics.thermodynamics import Thermodynamics
    from atmpy.grid.kgrid import Grid


class ClassicalPressureSolver(AbstractPressureSolver):
    """
    PressureSolver encapsulates the pressure correction procedure.
    It assembles the operator for the pressure correction (using, for example,
    a discrete Laplacian), builds the right–hand side from the divergence and other
    source terms, and then solves the resulting linear system via an injected linear solver.

    After obtaining the pressure correction, it updates the pressure (or p2-like)
    diagnostic in the Variables object. In a complete implementation the solver would
    also update ghost node values via boundary routines.
    """

    def __init__(
        self,
        discrete_operator: "AbstractDiscreteOperator",
        linear_solver: "ILinearSolver",
        grid: "Grid",
        variables: "Variables",
        mpv: "MPV",
        boundary_manager: "BoundaryManager",
        coriolis: "CoriolisOperator",
        thermodynamics: "Thermodynamics",
        Msq: float,
    ):
        super().__init__(
            discrete_operator, linear_solver, coriolis, thermodynamics, Msq
        )
        self.grid = grid
        self.variables: "Variables" = variables
        self.mpv: "MPV" = mpv
        self.boundary_manager: "BoundaryManager" = boundary_manager
        self.ndim = self.variables.ndim
        self.vertical_momentum_index: int = self.coriolis.gravity.gravity_momentum_index

    def pressure_coefficients_nodes(self, cellvars: np.ndarray, dt: float):
        """Calculate the coefficients for the pressure equation. Notice the coefficients are nodal.

        Basically it calculates there are two sets of coefficients needed to be calculated:
        1. alpha_h*(P*Theta) in Momentum equations (Coefficient of pressure term in Helmholtz equation)
        2. alpha_p*(dP/dpi) in Pressure equation   (Coefficient of pressure term in Momentum equation)

        The first will be stored in mpv.wplus
        The second will be stored in mpv.wcenter

        Parameters
        ----------
        cellvars: np.ndarray
            The array of variables on cells.
        dt: float
            The time step.

        Notes
        -----
        1. A variable container is passed to the method to decouple it from the attribute variable to be able to use the
            method on the initial variables in the trapezoidal rule.

        2. Two minor differences in calculation of alpha_p*(dP/dpi):

        - First, it considers that the alpha_p = 1, since for incompressible case, we handle the case elsewhere in the
        time update.
        - Second, it calculates this term for the Helmholtz equation of pressure. Therefore, it calculates the term after
        bringing dP/dpi to the right hand side of the Helmholtz equation for pressure:

        Remember the P and pi are connected to each other through the following formula:
        pi = (1/Msq) * P^(gamma - 1). Therefore, dpi/dP = (gamma - 1)/Msq * P^(gamma - 2). Since we need the inverse
        (dP/dpi), every part of this will be inverted in the code. The -dt**2 is due to the right-hand side of the
        Helmholtz equation."""

        #################### Calculate the coefficients ###############################################################
        pTheta = self._calculate_coefficient_pTheta(cellvars)  # Cell-centered
        dPdpi = self._calculate_coefficient_dPdpi(cellvars, dt)  # Node-centered

        ######### Fill wplux and wcenter containers with the corresponding values above ###############################
        for dim in range(self.ndim):
            self.mpv.wplus[dim][...] = pTheta

        inner_slice = one_element_inner_slice(self.ndim, full=False)
        self.mpv.wcenter[inner_slice] = dPdpi

        #################### Update the boundary nodes for the dP/dpi container. ######################################
        # Create the operation context to scale down the nodes. Notice the side is set to be BdrySide.ALL.
        # This will apply the 'extra' method whenever the boundary is defined to be WALL.
        boundary_operation = [
            WallAdjustment(
                target_side=BdrySide.ALL, target_type=BdryType.WALL, factor=0.5
            )
        ]
        self.boundary_manager.apply_extra_all_sides(
            self.mpv.wcenter, boundary_operation
        )

    def calculate_enthalpy_weighted_pressure_gradient(
        self, p: np.ndarray, dt: float, is_nongeostrophic: bool, is_nonhydrostatic: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the M^(-1)*dt*pTheta*grad(pi), where M is the extended coriolis matrix. The divergence of this term
        is the isentropic laplacian part of the Helmholtz operator. The whole operation takes place on cells.

        Parameters
        ----------
        p : np.ndarray
            The nodal pressure vector (or perturbation). The gradient bring this to cells.
        dt : float
            The time step
        is_nongeostrophic : bool
            The switch between geostrophic and non-geostrophic regimes
        is_nonhydrostatic : bool
            The switch between hydrostatic and non-hydrostatic regimes

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Tuple of updated momenta arrays
        """
        # Get necessary variables/coefficients
        cellvars = self.variables.cell_vars

        #### Calculate the needed values for the updates in the RHS of the momenta equations (the Exner pressure #######
        #### perturbation (Pi^prime)_x, _y, _z (see RHS of momenta eq.) and the pTheta coefficients)  ##################
        dpdx, dpdy, dpdz = self.discrete_operator.gradient(p)
        pTheta = self._calculate_coefficient_pTheta(cellvars)

        ##################### Calculate initial flux increment (cell-centered) #########################################
        u = -dt * pTheta * dpdx
        v = -dt * pTheta * dpdy
        w = -dt * pTheta * dpdz if self.ndim == 3 else np.zeros_like(dpdz)

        ##################### Apply Coriolis/Buoyancy transform (M_inv) ################################################
        self.coriolis.apply_inverse(
            u,
            v,
            w,
            self.variables,
            self.mpv,
            is_nongeostrophic,
            is_nonhydrostatic,
            self.Msq,
            dt,
        )

        return u, v, w

    def correction_nodes(
        self,
        p: np.ndarray,
        updt_chi: Union[np.ndarray, float],
        dt: float,
        is_nongeostrophic: bool,
        is_nonhydrostatic: bool,
    ):
        """Update the momenta and Chi variables using the pressure term with their coefficients on the RHS of the
        momenta equation.

        Parameters
        ----------
        p : np.ndarray
            The pressure vector. Basically placeholder for the perturbation of the Exner pressure (pi')
        updt_chi : np.ndarray
            The new updated value for Chi
        dt : float
            The time step
        is_nongeostrophic : bool
            The switch between geostrophic and non-geostrophic regimes
        is_nonhydrostatic : bool
            The switch between hydrostatic and non-hydrostatic regimes
        """

        ################################ Calculate the RHS of momenta equations  #######################################
        u_incr, v_incr, w_incr = self.calculate_enthalpy_weighted_pressure_gradient(
            p, dt, is_nongeostrophic, is_nonhydrostatic
        )

        ################################# Create the necessary variables  ##############################################
        cellvars = self.variables.cell_vars
        chi = cellvars[..., VI.RHO] / cellvars[..., VI.RHOY]
        dS = self.mpv.compute_dS_on_nodes()  # Assuming cell-centered dS/dy

        ########################## Update the full variables using the intermediate variables. #########################
        cellvars[..., VI.RHOU] += chi * u_incr
        cellvars[..., VI.RHOV] += chi * v_incr if self.ndim == 2 else 0.0
        cellvars[..., VI.RHOW] += chi * w_incr if self.ndim == 3 else 0.0
        cellvars[..., VI.RHOX] += (
            -updt_chi * dt * dS * cellvars[..., self.vertical_momentum_index]
        )

    def isentropic_laplacian(
        self,
        p: np.ndarray,
        dt: float,
        is_nongeostrophic: bool,
        is_nonhydrostatic: bool,

    ):
        """Compute the isentropic laplacian operator:  -∇⋅( M_inv ⋅ ( dt * (PΘ)° * ∇p ) )

        Parameters
        ----------
        p : np.ndarray
            The input Exner pressure perturbation. This is the scalar field on which the laplacian is computed.
        dt : float
            The time step
        is_nongeostrophic : bool
            The switch between geostrophic and non-geostrophic regimes
        is_nonhydrostatic : bool
            The switch between hydrostatic and non-hydrostatic regimes

        Returns
        -------
        np.ndarray (flattened)
            The isentropic laplacian operator

        Notes
        -----
        In order to pass the result to the linear solver (which only takes flattened input), we flatten the ONE ELEMENT
        INTERNAL nodes. That means we ignore one layer of ghost cells in each direction and then flatten the output.
        Therefore, notice the shapes:
        - p is of shape (nx+1, ny+1, nz+1)
        - grad(p) is of shape (nx, ny, nz)
        - divergence(grad(p)) is of shape (nx-1, ny-1, nz-1)
        """

        ######### Calculate the needed term inside the divergence: M_inv ⋅ ( dt * (PΘ)° * ∇p )  ########################
        # The results are cell-centered
        u, v, w = self.calculate_enthalpy_weighted_pressure_gradient(p, dt, is_nongeostrophic, is_nonhydrostatic)

        ######### Stack the values above so that they can be passed to the divergence ##################################
        vector_field_cell = np.stack(
            [u, v, w], axis=-1
            )[..., :self.ndim] # Ensure correct dims

        ######### Apply divergence #####################################################################################
        divergence = self.discrete_operator.divergence(vector_field_cell)

        ######### Negate the result and flatten for future usage in scipy LinearOperator ###############################
        laplacian = -divergence
        inner_slice = laplacian_inner_slice(self.grid.ng)
        result_flat = laplacian[inner_slice].flatten()

        return result_flat

    def helmholtz_operator(
        self,
        p: np.ndarray,
        dt: float,
        is_nongeostrophic: bool,
        is_nonhydrostatic: bool,
        is_compressible: bool,
    ):
        """ Calculate the full Helmholtz operator:
        [α_p * (∂P/∂π)°] * pi + IsentropicLaplacian(pi)

        Parameters
        ----------
        p : np.ndarray
            The input Exner pressure perturbation. This is the scalar field on which the laplacian is computed.
        dt : float
            The time step
        is_nongeostrophic : bool
            The switch between geostrophic and non-geostrophic regimes
        is_nonhydrostatic : bool
            The switch between hydrostatic and non-hydrostatic regimes
        is_compressible : bool
            The switch between compressible and incompressible regimes
        """
        ############### Calculate the Laplacian ########################################################################
        laplacian_flat = self.isentropic_laplacian(p, dt, is_nongeostrophic, is_nonhydrostatic)


        ####### Creating the pressure term corresponding to the dPdpi and add it to the laplacian ######################
        inner_slice = tuple(self.grid.inner_slice)
        if is_compressible:
            wcenter_flat = self.mpv.wcenter[inner_slice].flatten()
            helmholtz_result_flat = laplacian_flat + wcenter_flat * p[inner_slice].flatten()
        else:
            helmholtz_result_flat = laplacian_flat

        return helmholtz_result_flat

    def _calculate_P_over_Gamma(self, cellvars: np.ndarray):
        """Calculates P/Gamma. This is an intermediate function to avoid duplicate codes.

        Parameters
        ----------
        cellvars : np.ndarray
            The full variable container for cell-centered variables
        th: Thermodynamics
            The object of thermodynamics class

        """
        return self.th.Gammainv * cellvars[..., VI.RHOY]

    def _calculate_coefficient_pTheta(self, cellvars: np.ndarray):
        """First part of coefficient calculation. Calculates P*Theta."""

        # Calculate (P*Theta): Coefficient of pressure term in momentum equation
        Y = cellvars[..., VI.RHOY] / cellvars[..., VI.RHO]
        return self._calculate_P_over_Gamma(cellvars) * Y

    def _calculate_coefficient_dPdpi(self, cellvars: np.ndarray, dt: float):
        """Calculate the second part of the coefficient calculation. Calculate dP/dpi. See docstring of
        operator_coefficients_nodes for more information.

        Notes
        -----
        The shape of the output is the same as the inner NODES, which incidentally (but evidently) is equal to the cell
        shape of the grid (cshape)
        """

        # Calculate the coefficient and the exponent of the dP/dpi using the formula directly. (see the docstring)
        ccenter = -self.Msq * self.th.gm1inv / (dt ** 2)
        cexp = 2.0 - self.th.gamma

        # Temp variable for rhoTheta=P for readability
        P = cellvars[..., VI.RHOY]

        # Averaging over the nodes and fill the mpv container
        kernel = np.ones([2] * self.ndim)
        return (
                ccenter
                * sp.signal.fftconvolve(P ** cexp, kernel, mode="valid")
                / kernel.sum()
        )