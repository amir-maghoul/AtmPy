"""This module contains different time integrators"""

from __future__ import annotations
from typing import TYPE_CHECKING, Union, List

from atmpy.boundary_conditions.bc_extra_operations import WallAdjustment

if TYPE_CHECKING:
    from atmpy.flux.flux import Flux
    from atmpy.variables.variables import Variables
    from atmpy.variables.multiple_pressure_variables import MPV
    from atmpy.grid.kgrid import Grid
    from atmpy.boundary_conditions.boundary_manager import BoundaryManager
    from atmpy.physics.gravity import Gravity
    from atmpy.physics.thermodynamics import Thermodynamics

from atmpy.time_integrators.abstract_time_integrator import AbstractTimeIntegrator
from atmpy.time_integrators.utility import nodal_derivative, nodal_gradient, pressured_momenta_divergence
import scipy.sparse.linalg
from atmpy.infrastructure.enums import BoundarySide as BdrySide, BoundaryConditions as BdryType
from atmpy.time_integrators.utility import *



class IMEXTimeIntegrator(AbstractTimeIntegrator):
    def __init__(
        self,
        grid: "Grid",
        variables: "Variables",
        mpv: "MPV",
        flux: "Flux",
        boundary_manager: "BoundaryManager",
        coriolis_operator: CoriolisOperator,
        pressure_solver: PressureSolver,
        thermodynamics: "Thermodynamics",
        dt: float,
        **kwargs
    ):
        # Inject dependencies
        self.grid: "Grid" = grid
        self.variables: "Variables" = variables
        self.mpv: "MPV" = mpv
        self.flux: "Flux" = flux
        self.boundary_manager: "BoundaryManager" = boundary_manager
        self.coriolis: "CoriolisOperator" = coriolis_operator
        self.gravity: "Gravity" = self.coriolis.gravity
        self.pressure_solver: "PressureSolver" = pressure_solver
        self.th: "Thermodynamics" = thermodynamics
        self.dt: float = dt
        self.wind_speed: Union[list, np.ndarray] = kwargs.get("wind_speed")


    def step(self):
        # 1. Explicit forward update (e.g. divergence, pressure gradient, momentum update)
        self.forward_update()

        # 2. Apply boundary conditions (if needed)
        self.boundary_manager.apply_all(self.variables)

        # 3. Implicit backward update: solve for the pressure correction
        self.backward_update_implicit()

        # 4. Explicit correction update (e.g. adjusting momentum with background winds, applying inverse Coriolis)
        self.backward_update_explicit()

        # 5. Final boundary condition update, if needed.
        self.boundary_manager.apply_all(self.variables)

    def forward_update(self):
        """ Integrates the problem on time step using the explicit euler for non-advective terms."""
        g = self.gravity.strength
        if self.wind_speed:
            self.variables.adjust_background_wind(self.wind_speed, -1.0)

        # second pressure asymptotics on nodes
        p2n = np.copy(self.mpv.p2_nodes)

        # container for the derivative of p2 on nodes.
        dp2n = np.zeros_like(p2n)

        # get Chi variable and the derivative
        S0c = self.mpv.get_S0c_on_cells()
        dSdy = self.mpv.compute_dS_on_nodes(self.gravity.direction)

        # Calculate the divergence of momenta on the NODES. This is the right hand side of the pressure equation
        self.mpv.rhs[...] = pressured_momenta_divergence(self.grid, self.variables)

        # Adjust wall boundary nodes (scale). Notice the side is set to be BdrySide.ALL.
        # This will apply the extra method whenever the boundary is defined to be WALL.
        boundary_operation = [WallAdjustment(target_side=BdrySide.ALL, target_type=BdryType.WALL, factor=2.0)]
        self.boundary_manager.apply_extra_all_sides(self.mpv.rhs, boundary_operation)

        # index 0: momentum in the direction of gravity. index 1: momentum in the direction of non-gravity.
        momentum_index = self.gravity.momentum_index

        #



    def backward_update_implicit(self):
        # Encapsulate the pressure solve (euler_backward_non_advective_impl_part)
        # For example, assemble the pressure operator and call the pressure_solver.
        self.pressure_solver.solve(self.variables, self.dt)
        print("Implicit (pressure) update complete.")

    def backward_update_explicit(self):
        # Encapsulate the explicit corrections like background wind modifications
        # and the explicit multiplicative inverse of the Coriolis operator
        # (euler_backward_non_advective_expl_part)
        self.adjust_background_wind(-1.0)
        self.coriolis_operator.apply(self.variables, self.dt)
        self.adjust_background_wind(+1.0)
        print("Explicit backward update complete.")

    def get_dt(self):
        return self.dt


class CoriolisOperator:
    """
    Encapsulates the handling of coriolis operator in the implicit time update. This is a part of operator splitting
    for stiff coriolis operator.
    """

    def __init__(
        self,
        coriolis_strength: Union[np.ndarray, list],
        gravity: "Gravity",
        Msq: float,
        nonhydro: bool,
        get_strat: callable,
    ):
        self.coriolis_strength: Union[np.ndarray, list] = coriolis_strength
        self.gravity: "Gravity" = gravity
        self.Msq: float = Msq
        self.nonhydro: bool = nonhydro
        self.get_strat: callable = get_strat  # callable that returns stratification

    def apply(self, variables: "Variables", dt: float) -> None:
        """Apply correction to momenta due to the coriolis effect. If there are no coriolis forces in any direction,
        do nothing.

        Parameters
        ----------
        variables : Variables
            The variable container containing the momenta.
        dt : float
            The time step.
        """
        if self.coriolis_strength is None or np.all(self.coriolis_strength == 0):
            pass
        else:
            self._apply(variables, dt)

    def _apply(self, variables: "Variables", dt: float):
        # dt-scaled Coriolis parameters
        wh1, wv, wh2 = dt * np.array(self.coriolis_strength)
        # Obtain stratification (here, a dummy constant field)
        strat = self.get_strat(variables)
        Y = variables.rhoY / variables.rho
        nu = -(dt**2) * (self.gravity_strength[1] / self.Msq) * strat * Y

        # Common denominator simulating inversion of (I+dt C)
        denom = 1.0 / (wh1**2 + wh2**2 + (nu + self.nonhydro) * (wv**2 + 1.0))
        # Compute cross-coupling coefficients
        coeff_uu = wh1**2 + nu + self.nonhydro
        coeff_uv = self.nonhydro * (wh1 * wv + wh2)
        coeff_uw = wh1 * wh2 - (nu + self.nonhydro) * wv

        coeff_vu = wh1 * wv - wh2
        coeff_vv = self.nonhydro * (1 + wv**2)
        coeff_vw = wh2 * wv + wh1

        # Only update W if in 3D
        if variables.rhow is not None:
            coeff_wu = wh1 * wh2 + (nu + self.nonhydro) * wv
            coeff_wv = self.nonhydro * (wh2 * wv - wh1)
            coeff_ww = nu + self.nonhydro + wh2**2

        # Copy old momentum fields
        U_old = variables.rhou.copy()
        V_old = variables.rhov.copy()
        if variables.rhow is not None:
            W_old = variables.rhow.copy()
        else:
            W_old = 0.0

        # Update momentum fields by “inverting” the Coriolis source.
        variables.rhou[...] = denom * (
            coeff_uu * U_old + coeff_uv * V_old + (coeff_uw * W_old)
        )
        variables.rhov[...] = denom * (
            coeff_vu * U_old + coeff_vv * V_old + (coeff_vw * W_old)
        )
        if variables.rhow is not None:
            variables.rhow[...] = denom * (
                coeff_wu * U_old + coeff_wv * V_old + coeff_ww * W_old
            )
        print("Coriolis operator applied.")


class PressureSolver:
    """
    PressureSolver encapsulates the pressure correction procedure.
    It assembles the operator for the pressure correction (using, for example,
    a discrete Laplacian), builds the right–hand side from the divergence and other
    source terms, and then solves the resulting linear system via an injected linear solver.

    After obtaining the pressure correction, it updates the pressure (or p2-like)
    diagnostic in the Variables object. In a complete implementation the solver would
    also update ghost node values via boundary routines.
    """

    def __init__(self, linear_solver):
        """
        linear_solver: an instance implementing the ILinearSolver interface
                       (e.g., a BiCGStabSolver).
        """
        self.linear_solver = linear_solver

    def build_operator(self, variables, grid, dt):
        """
        Assemble the pressure correction operator as a linear operator.
        This might be built using a multigrid, 9-point, or 27-point stencil, for example.

        For this simplified example we assume a 2D discrete Laplacian over the grid.
        """
        shape = (
            variables.rho.shape
        )  # assume the same shape for the diagnostic pressure field
        size = np.prod(shape)

        # Define the operator's action (a dummy Laplacian for illustration)
        def matvec(x):
            # x comes in as a flattened vector. Reshape to grid form.
            u = x.reshape(shape)
            laplacian = np.zeros_like(u)
            # A simple finite-difference Laplacian for interior points:
            laplacian[1:-1, 1:-1] = (
                u[:-2, 1:-1] - 2 * u[1:-1, 1:-1] + u[2:, 1:-1]
            ) / grid.dx**2 + (
                u[1:-1, :-2] - 2 * u[1:-1, 1:-1] + u[1:-1, 2:]
            ) / grid.dy**2
            # (In practice, ghost cells and boundary conditions are important.)
            return laplacian.ravel()

        A = scipy.sparse.linalg.LinearOperator((size, size), matvec=matvec)
        return A

    def build_rhs(self, variables, grid, dt):
        """
        Build the right-hand side (RHS) for the pressure correction.
        Typically this uses the divergence of the momentum field along with
        other contributions (for instance, compressibility or additional source terms).

        In the legacy code, functions such as divergence_nodes and adjustments for
        compressibility are used. Here we provide a simplified placeholder version.
        """
        shape = variables.rho.shape
        rhs = np.zeros(shape)

        # For example, compute a simple discrete divergence based on momentum differences.
        # (Replace this with a proper discrete operator as needed.)
        rhs[1:-1, 1:-1] = (variables.rhou[2:, 1:-1] - variables.rhou[:-2, 1:-1]) / (
            2 * grid.dx
        ) + (variables.rhov[1:-1, 2:] - variables.rhov[1:-1, :-2]) / (2 * grid.dy)
        # Additional modifications (e.g. compressibility weighting) may be applied.

        # Flatten rhs for the linear solver.
        return rhs.ravel()

    def solve(self, variables, grid, dt):
        """
        Perform the pressure correction step:
          1. Build the operator A for the pressure correction.
          2. Compute the right-hand side vector from the momentum divergence and sources.
          3. Solve the linear system A * p_corr = rhs with the injected linear solver.
          4. Reshape and update the diagnostic pressure field (e.g., p2_nodes or variables.pressure).

        Boundary updates and ghost cell enforcement can be handled here or delegated.
        """
        A = self.build_operator(variables, grid, dt)
        rhs = self.build_rhs(variables, grid, dt)

        # Solve the system (using, e.g., BiCGStab)
        solution = self.linear_solver.solve(A, rhs)
        # Reshape the solution to the grid dimensions.
        pressure_correction = solution.reshape(variables.rho.shape)

        # Update the pressure diagnostic field.
        # In the legacy code, this update might look like:
        #   mpv.p2_nodes += weight * dp2n  (with dp2n computed via pressure derivative kernels)
        try:
            # If the field already exists, update it.
            variables.pressure += pressure_correction * dt
        except AttributeError:
            # Otherwise, create it as a new field.
            variables.pressure = pressure_correction * dt

        # One might update boundary ghost values here as well,
        # e.g., boundary_manager.set_ghostnodes_pressure(variables.pressure, node, user_data)
        print("Pressure solve complete and pressure field updated.")

if __name__ == '__main__':
    pass