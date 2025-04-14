"""This module handles solving the equation for the pressure variable including the Laplace/Poisson equation."""

from abc import ABC, abstractmethod
import numpy as np
import scipy as sp
from typing import TYPE_CHECKING, Union, TypeVar

from atmpy.infrastructure.enums import VariableIndices as VI, BoundarySide as BdrySide, BoundaryConditions as BdryType
from atmpy.boundary_conditions.bc_extra_operations import WallAdjustment
from atmpy.infrastructure.utility import one_element_inner_slice
from atmpy.physics import thermodynamics
from atmpy.physics.thermodynamics import Thermodynamics

if TYPE_CHECKING:
    from atmpy.pressure_solver.discrete_operations import AbstractDiscreteOperator
    from atmpy.pressure_solver.linear_solvers import ILinearSolver
    from atmpy.variables.variables import Variables
    from atmpy.variables.multiple_pressure_variables import MPV
    from atmpy.boundary_conditions.boundary_manager import BoundaryManager
    from atmpy.time_integrators.coriolis import CoriolisOperator
    from atmpy.physics.thermodynamics import Thermodynamics


class AbstractPressureSolver(ABC):
    """Create different definitions of pressure solvers."""

    def __init__(
        self,
        discrete_operator: "AbstractDiscreteOperator",
        linear_solver: "ILinearSolver",
        coriolis: "CoriolisOperator",
        thermodynamics: "Thermodynamics",
        Msq: float,
        dt: float,
    ):
        self.discrete_operator: "AbstractDiscreteOperator" = discrete_operator
        self.linear_solver: "ILinearSolver" = linear_solver
        self.coriolis: "CoriolisOperator" = coriolis
        self.th: "Thermodynamics" = thermodynamics
        self.Msq: float = Msq
        self.dt: float = dt
        self.vertical_momentum_index: int = self.coriolis.gravity.gravity_momentum_index

TPressureSolver = TypeVar('TPressureSolver', bound=AbstractPressureSolver)

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
        variables: "Variables",
        mpv: "MPV",
        boundary_manager: "BoundaryManager",
        coriolis: "CoriolisOperator",
        thermodynamics: "Thermodynamics",
        is_compressible: bool,
        is_nongeostrophic: bool,
        is_nonhydrostatic: bool,
        Msq: float,
        dt: float,

    ):
        super().__init__(discrete_operator, linear_solver, coriolis, thermodynamics, Msq, dt)
        self.variables: "Variables" = variables
        self.mpv: "MPV" = mpv
        self.boundary_manager: "BoundaryManager" = boundary_manager
        self.ndim = self.variables.ndim
        self.is_compressible: bool = is_compressible
        self.is_nongeostrophic: bool = is_nongeostrophic
        self.is_nonhydrostatic: bool = is_nonhydrostatic
        self.vertical_momentum_index: int = self.coriolis.gravity.gravity_momentum_index


    def pressure_coefficients_nodes(self, cellvars: np.ndarray):
        """ Calculate the coefficients for the pressure equation. Notice the coefficients are nodal.

        Basically it calculates there are two sets of coefficients needed to be calculated:
        1. alpha_h*(P*Theta) in Momentum equations (Coefficient of pressure term in Helmholtz equation)
        2. alpha_p*(dP/dpi) in Pressure equation   (Coefficient of pressure term in Momentum equation)

        The first will be stored in mpv.wplus
        The second will be stored in mpv.wcenter

        Parameters
        ----------
        cellvars: np.ndarray
            The array of variables on cells.

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
        pTheta = self._calculate_coefficient_pTheta(cellvars)
        dPdpi = self._calculate_coefficient_dPdpi(cellvars)

        ######### Fill wplux and wcenter containers with the corresponding values above ###############################
        for dim in range(self.ndim):
            self.mpv.wplus[dim][...] = pTheta

        self.mpv.wcenter = dPdpi

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

    def correction_nodes(self, p: np.ndarray, updt_chi: Union[np.ndarray, float]):
        """ Adjust the momenta and Chi variables.

        Parameters
        ----------
        p : np.ndarray
            The pressure vector. Basically placeholder for the perturbation of the Exner pressure (pi')
        updt_chi : np.ndarray
            The new updated value for Chi
        """

        # Temporary variable for readability
        cellvars = self.variables.cell_vars
        dS = self.mpv.compute_dS_on_nodes()

        #### Calculate the needed values for the updates in the RHS of the momenta equations (the Exner pressure
        #### perturbation (Pi^prime)_x, _y, _z (see RHS of momenta eq.) and the pTheta coefficients)
        dpdx, dpdy, dpdz = self.discrete_operator.gradient(p)
        pTheta = self._calculate_coefficient_pTheta(cellvars)

        #### calculate the intermediate variables
        u = -self.dt * pTheta * dpdx
        v = -self.dt * pTheta * dpdy
        w = -self.dt * pTheta * dpdz

        #### Adjust the intermediate variables using the coriolis update
        self.coriolis.apply(
            u,
            v,
            w,
            self.variables,
            self.mpv,
            self.is_nongeostrophic,
            self.is_nonhydrostatic,
            self.Msq,
            self.dt,
        )

        ### Update the full variables using the intermediate variables.
        chi = cellvars[..., VI.RHO] / cellvars[..., VI.RHOY]
        cellvars[..., VI.RHOU] += chi * self.mpv.u
        cellvars[..., VI.RHOV] += chi * self.mpv.v if self.ndim == 2 else 0.0
        cellvars[..., VI.RHOW] += chi * self.mpv.w if self.ndim == 3 else 0.0
        cellvars[..., VI.RHOX] += -updt_chi * self.dt * dS * cellvars[..., self.vertical_momentum_index]

    def rhs_from_p_old(self):
        """ Calculate the first time on RHS of the pressure equation: dP/dpi * pi^{prime} """

        # Create node-to-cell index (slice(1, -1) in all directions)
        inner_idx = one_element_inner_slice(self.ndim, full=False)

        # Remember: wcenter is container for dP/dpi and p2_nodes is Exner perturbation
        return self.mpv.wcenter * self.mpv.p2_nodes[inner_idx]

    def _calculate_coefficient_pTheta(self, cellvars: np.ndarray):
        """First part of coefficient calculation. Calculates P*Theta. See docstring of operator_coefficients_nodes for
        more information."""

        # Calculate (P*Theta): Coefficient of pressure term in momentum equation
        Y = (
                cellvars[..., VI.RHOY]
                / cellvars[..., VI.RHO]
        )
        return self._calculate_P_over_Gamma(cellvars) * Y

    def _calculate_P_over_Gamma(self, cellvars):
        """ Calculates P/Gamma. This is an intermediate function to avoid duplicate codes."""
        return self.th.Gammainv * cellvars[..., VI.RHOY]

    def _calculate_coefficient_dPdpi(self, cellvars: np.ndarray):
        """Calculate the second part of the coefficient calculation. Calculate dP/dpi. See docstring of
        operator_coefficients_nodes for more information."""

        # Calculate the coefficient and the exponent of the dP/dpi using the formula directly. (see the docstring)
        ccenter = -self.Msq * self.th.gm1inv / (self.dt ** 2)
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

    def solve(self, variables, grid, dt):
        pass



