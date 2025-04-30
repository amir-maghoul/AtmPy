"""This module contains different time integrators"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Union, List, Any, Callable


from atmpy.boundary_conditions.bc_extra_operations import WallAdjustment
from atmpy.boundary_conditions.contexts import BCApplicationContext
from atmpy.infrastructure.utility import (
    directional_indices,
    one_element_inner_slice,
    one_element_inner_nodal_shape,
)
from atmpy.pressure_solver.discrete_operations import AbstractDiscreteOperator
from atmpy.infrastructure.enums import VariableIndices as VI, Preconditioners

if TYPE_CHECKING:
    from atmpy.flux.flux import Flux
    from atmpy.variables.variables import Variables
    from atmpy.variables.multiple_pressure_variables import MPV
    from atmpy.grid.kgrid import Grid
    from atmpy.boundary_conditions.boundary_manager import BoundaryManager
    from atmpy.physics.gravity import Gravity
    from atmpy.physics.thermodynamics import Thermodynamics
    from atmpy.time_integrators.coriolis import CoriolisOperator
    from atmpy.pressure_solver.abstract_pressure_solver import TPressureSolver

from atmpy.infrastructure.enums import AdvectionRoutines
from atmpy.pressure_solver.classical_pressure_solvers import ClassicalPressureSolver

from atmpy.time_integrators.abstract_time_integrator import AbstractTimeIntegrator
from atmpy.time_integrators.utility import *

from atmpy.infrastructure.enums import (
    BoundarySide as BdrySide,
    BoundaryConditions as BdryType,
)


class IMEXTimeIntegrator(AbstractTimeIntegrator):
    def __init__(
        self,
        grid: "Grid",
        variables: "Variables",
        mpv: "MPV",
        flux: "Flux",
        boundary_manager: "BoundaryManager",
        pressure_solver: "TPressureSolver",
        advection_routine: AdvectionRoutines,
        wind_speed: List[float],
        is_nongeostrophic: bool,
        is_nonhydrostatic: bool,
        is_compressible: bool,
        dt: float,
    ):
        # Inject dependencies
        self.grid: "Grid" = grid
        self.variables: "Variables" = variables
        self.mpv: "MPV" = mpv
        self.flux: "Flux" = flux
        self.pressure_solver: "TPressureSolver" = pressure_solver

        if not isinstance(self.pressure_solver, ClassicalPressureSolver):
            raise ValueError(
                "The current implementation of IMEXTimeIntegrator only supports ClassicalPressureSolver."
            )

        self.boundary_manager: "BoundaryManager" = boundary_manager
        self.coriolis: "CoriolisOperator" = self.pressure_solver.coriolis
        self.gravity: "Gravity" = self.coriolis.gravity
        self.discrete_operator: "AbstractDiscreteOperator" = (
            self.pressure_solver.discrete_operator
        )
        self.advection_routine: Callable
        self._get_advection_routine(advection_routine)
        self.th: "Thermodynamics" = self.pressure_solver.th
        self.dt: float = dt
        self.Msq: float = self.pressure_solver.Msq
        self.is_nongeostrophic: bool = is_nongeostrophic
        self.is_nonhydrostatic: bool = is_nonhydrostatic
        self.is_compressible: bool = is_compressible
        self.wind_speed: np.ndarray = np.array(wind_speed)
        self.ndim = self.grid.ndim
        self.vertical_momentum_index = self.coriolis.gravity.gravity_momentum_index

        # Helper for nodal boundary conditions
        # That is setting the flag is_nodal for all dimensions and all sides (therefore ndim*2) to True
        # since p2_nodes is nodal.
        self._nodal_bc_contexts = (
            [BCApplicationContext(is_nodal=True)] * self.grid.ndim * 2
        )

    def _get_advection_routine(
        self, advection_routine_name: "AdvectionRoutines"
    ) -> None:
        """Get the advection routine. The sole raison d'être of this method is to avoid circular import
        issues from factory."""
        from atmpy.infrastructure.factory import get_advection_routines

        self.advection_routine = get_advection_routines(advection_routine_name)


    def step(self) -> None:
        """
        Performs a single time step using the semi-implicit predictor-corrector
        method based on Benacchio & Klein (2019).
        """
        logging.info(f"--- Starting time step with dt = {self.dt} ---")

        # --- Save Initial State (t^n) ---
        initial_vars = np.copy(self.variables.cell_vars)
        initial_p2 = np.copy(self.mpv.p2_nodes)
        # It might be necessary to save self.flux too if advection modifies it undesirably
        # initial_flux_iflux = {k: np.copy(v) for k, v in self.flux.iflux.items() if v is not None}

        ####################### Predictor Stage (Calculates fluxes at t^{n+1/2}) #######################################
        # This modifies self.variables, self.mpv, and self.flux
        self._predictor_step(self.dt, self.advection_routine)

        ####################### Corrector Stage (Advances state from t^n to t^{n+1}) ###################################
        # This resets self.variables/mpv internally and uses the predicted fluxes from self.flux
        self._corrector_step(self.dt, self.advection_routine, initial_vars, initial_p2)

        logging.info(f"--- Finished time step ---")

    def _predictor_step(self, dt: float, advection_func: Callable) -> None:
        """
        Performs the predictor stage of the time step (Eq. 14-15 in BK19).
        Calculates the state at t + dt/2 to compute advective fluxes for the main step.
        Modifies self.variables, self.mpv, and self.flux in-place.
        """
        logging.debug("--- Starting Predictor Step ---")
        half_dt = 0.5 * dt

        ############################## 1. Apply BCs to current state ###################################################
        self.boundary_manager.apply_boundary_on_all_sides(self.variables.cell_vars)
        self.boundary_manager.apply_boundary_on_single_var_all_sides(
            self.mpv.p2_nodes, self._nodal_bc_contexts
        )

        ###################### 2. Compute advective fluxes based on state at t^n #######################################
        # Note: compute_averaging_fluxes computes Pu, Pv, Pw.
        # The Riemann solver part happens within advection_func.
        self.flux.compute_averaging_fluxes()

        ################### 3. Advect state by dt/2 using fluxes at t^n (Eq. 14a, but integrated) ######################
        # Result is Sol*
        advection_func(
            self.grid,
            self.variables,
            self.flux,
            half_dt,
            boundary_manager=self.boundary_manager,
        )
        logging.debug("Predictor: After advection")

        ################### 4. Save current pressure p^n before non-advective updates ##################################
        # (Needed if compressibility logic requires p2_nodes0 later, but BK19 C code resets p2_nodes later)
        # p2_nodes_n = np.copy(self.mpv.p2_nodes) # Let's assume implicit step correctly uses current p2

        ################### 5. Non-Advective Implicit Euler Substep for dt/2 (Eq. 15) ##################################
        # Result is Sol^{n+1/2} and p2^{n+1/2}
        self.backward_update_explicit(half_dt)
        logging.debug("Predictor: After backward_update_explicit")
        # Note: backward_update_implicit uses the state modified by backward_update_explicit
        self.backward_update_implicit(half_dt)
        logging.debug("Predictor: After backward_update_implicit")

        ################### 6. Compute predicted advective fluxes (Pu, Pv, Pw) at t^{n+1/2} ############################
        # These will be used by the corrector stage's advection step.
        # This updates self.flux.iflux and potentially parts of self.flux.flux
        self.flux.compute_averaging_fluxes()
        logging.debug("--- Finished Predictor Step (Predicted Fluxes Updated) ---")

    def _corrector_step(
        self,
        dt: float,
        advection_func: Callable,
        initial_vars: np.ndarray,
        initial_p2: np.ndarray,
    ) -> None:
        """
        Performs the corrector stage of the time step (Eq. 17 in BK19).
        Advances the state from t^n to t^{n+1} using predicted advective fluxes.
        """
        logging.debug("--- Starting Corrector Step ---")
        half_dt = 0.5 * dt

        ######################### 1. Reset state variables to t^n ######################################################
        self.variables.cell_vars[...] = initial_vars
        self.mpv.p2_nodes[...] = initial_p2
        logging.debug("Corrector: State reset to t^n")

        ######################### 2. Apply BCs to state at t^n #########################################################
        self.boundary_manager.apply_boundary_on_all_sides(self.variables.cell_vars)
        self.boundary_manager.apply_boundary_on_single_var_all_sides(
            self.mpv.p2_nodes, self._nodal_bc_contexts
        )

        ######################### 3. Explicit Predictor (Fast Modes, Eq. 17a) ##########################################
        # Applies explicit Euler step for non-advective terms based on state n
        # Result is Sol*
        self.forward_update(half_dt)
        logging.debug("Corrector: After forward_update (Eq. 17a)")

        ######################### 4. Advection (Full Step, Eq. 17b) ####################################################
        # Uses the predicted fluxes computed at the end of _predictor_step
        # (These are stored in self.flux)
        # Result is Sol**
        advection_func(
            self.grid,
            self.variables,
            self.flux,
            dt,
            boundary_manager=self.boundary_manager,
        )
        logging.debug("Corrector: After full advection (Eq. 17b)")

        ######################### 5. Implicit Corrector (Fast Modes, Eq. 17c) ##########################################
        # Applies implicit Euler substep for dt/2 to the advected state Sol**
        # Result is Sol^{n+1} and p2^{n+1}
        self.backward_update_explicit(half_dt)
        logging.debug("Corrector: After backward_update_explicit (part of Eq. 17c)")
        self.backward_update_implicit(half_dt)  # Uses state from previous step
        logging.debug("Corrector: After backward_update_implicit (part of Eq. 17c)")

        ######################## 6. Final Boundary Conditions ##########################################################
        self.boundary_manager.apply_boundary_on_all_sides(self.variables.cell_vars)
        self.boundary_manager.apply_boundary_on_single_var_all_sides(
            self.mpv.p2_nodes, self._nodal_bc_contexts
        )
        logging.debug("--- Finished Corrector Step ---")

    def forward_update(self, dt: float) -> None:
        """Integrates the problem on time step using the explicit euler in the NON-ADVECTIVE stage. This means
        that the algorithm does not use the half-time advective values to compute the updates. Rather, we compute
        directly from the full-variables in the main euler equation."""
        g = self.gravity.strength  # The gravity strength
        cellvars = self.variables.cell_vars  # Temp variable for cell variables
        p2n = np.copy(self.mpv.p2_nodes)  # Temp allocation for p2_nodes

        # Calculate the buoyancy PX'
        dbuoy = cellvars[..., VI.RHOY] * cellvars[..., VI.RHOX] / cellvars[..., VI.RHO]

        ###################### Update variables
        self._forward_momenta_update(cellvars, dt, p2n, dbuoy, g)
        self._forward_buoyancy_update(cellvars, dt)
        self._forward_pressure_update(cellvars, dt, p2n)

        ####################### Update boundary values
        # Update the boundary nodes for pressure variable
        self.boundary_manager.apply_boundary_on_single_var_all_sides(
            self.mpv.p2_nodes, self._nodal_bc_contexts
        )

        # Update all other variables on the boundary.
        self.boundary_manager.apply_boundary_on_all_sides(cellvars)

    def backward_update_explicit(self, dt: float):
        """This is the first part of implicit Euler update. This method does the job to calculate the terms involving the
        n-th timestep in the implicit scheme. The method backward_implicit_update involves the terms evaluated at the
        (n+1)-th timestep in the implicit scheme.

        Parameters
        ----------
        dt: float
            The time step used for the update at this stage. It could be a full global step or half global step.
        """

        cellvars = self.variables.cell_vars

        # First calculate the extra explicit buoyancy term that is not calculated in the coriolis matrix inversion:
        bouyoncy = cellvars[..., VI.RHOY] * (
            cellvars[..., VI.RHOX] / cellvars[..., VI.RHO]
        )

        # Update the corresponding vertical momentum explicitly
        g = self.gravity.strength

        if self.is_nonhydrostatic:
            cellvars[..., self.vertical_momentum_index] -= dt * (
                (g / self.Msq) * bouyoncy
            )

        # Remove background wind
        self.variables.adjust_background_wind(
            self.wind_speed, scale=-1.0, in_place=True
        )

        # Apply the solver inverse matrix (Matrix combining the switches, the coriolis force and the singular buoyancy term)
        self.coriolis.apply_inverse(
            cellvars[..., VI.RHOU],
            cellvars[..., VI.RHOV],
            cellvars[..., VI.RHOW],
            self.variables,
            self.mpv,
            self.is_nongeostrophic,
            self.is_nonhydrostatic,
            self.Msq,
            dt,
        )

        # Restore background wind
        self.variables.adjust_background_wind(self.wind_speed, scale=1.0, in_place=True)

        # Update all other variables on the boundary.
        self.boundary_manager.apply_boundary_on_all_sides(cellvars)

    def backward_update_implicit(self, dt: float, initial_vars: np.ndarray = None):
        """Compute the one step of the implicit part of the Euler backward scheme. This is a part of the BK19 algorithm.
        Notice before the call to this method, the coefficient variables must be created at half timestep. Then going back
        to the initial variables, we start anew to advance the time stepping in a implicit trapezoidal rule

        Parameters
        ----------
        dt: float
            The time step used for this update. It could be a full global step or half global step.
        initial_vars: np.ndarray
            The initial variables of the problem before any changes were made by calculation of half-time step.
        """
        cellvars = self.variables.cell_vars if initial_vars is None else initial_vars

        ################################ 1. Preparation ################################################################
        # Update the boundary value from current variables/initial variable and compute the pressure coefficients
        # The coefficient values depend on the current values in the variables. Therefore, they are pre update coefficients
        self.boundary_manager.apply_boundary_on_all_sides(cellvars)
        self.pressure_solver.pressure_coefficients_nodes(cellvars, dt)

        ################################ 2. Apply boundary conditions on pi' ###########################################
        # Update the boundary nodes for pressure variable
        self.boundary_manager.apply_boundary_on_single_var_all_sides(
            self.mpv.p2_nodes, self._nodal_bc_contexts
        )

        ################################ 3. "Pre-Correction" using Current Pressure p^k ################################
        # This method will put M_inv⋅(dt*CΘ*∇p₂)/Θ in the momenta container, where M is extended coriolis inverse.
        # In a couple of steps the divergence will be applied on [Pu, Pv, Pw], using these values.
        # The buoyancy term is NOT adjusted in this preparatory step (updt_chi=0.0). Only the momenta is needed for div.
        CHI_UPDT_VALUE = 0.0
        self.pressure_solver.apply_pressure_gradient_update(
            p=self.mpv.p2_nodes,  # Use current pressure
            updt_chi=CHI_UPDT_VALUE,  # Don't adjust buoyancy here
            dt=dt,
            is_nongeostrophic=self.is_nongeostrophic,
            is_nonhydrostatic=self.is_nonhydrostatic,
        )

        ############################# 4. Apply BCs to Pre-Corrected State ##############################################
        # TODO: Check if necessary
        self.boundary_manager.apply_boundary_on_all_sides(self.variables.cell_vars)

        ############################# 5. Compute RHS (Divergence of Pre-Corrected Momenta) #############################
        # Get the pre-corrected variables
        cellvars = self.variables.cell_vars

        # Calculate pressure-weighted momenta P*v = (rhoY/rho) * rho*v
        # [Pu, Pv, Pw]
        pressure_weighted_momenta = self._calculate_enthalpy_weighted_momenta(cellvars)

        # Calculate divergence on one element inner nodes [shape: (nx-1, ny-1, nz-1)]
        # Since the current momenta contain the pressure gradient update, this divergence
        # is basically ∇⋅(M_inv⋅(dt*(PΘ)*∇p₂)) where M is extended coriolis inverse
        divergence_inner = self.discrete_operator.divergence(pressure_weighted_momenta)

        # Final RHS for A * delta_p = rhs
        rhs_flat = divergence_inner.flatten()

        # Adjust the coefficients of the pressure gradient term for the solver as a result of compressibility regime
        self.mpv.wcenter *= self.is_compressible

        ############################ 6. Solve the elliptic helmholtz equation ##########################################
        p2_inner_flat, solver_info = self.pressure_solver.solve_helmholtz(
            rhs_flat,
            dt,
            self.is_nongeostrophic,
            self.is_nonhydrostatic,
            self.is_compressible,
            # Add tol/max_iter if needed, e.g., tol=1e-7
        )

        ############################ 7. Prepare p2 for Correction Step #################################################
        inner_slice = one_element_inner_slice(self.ndim, full=False)
        inner_shape = one_element_inner_nodal_shape(self.grid.nshape)
        p_unflat = p2_inner_flat.reshape(inner_shape)

        # Pad increment to full nodal shape
        p2_full = np.zeros_like(self.mpv.p2_nodes)
        p2_full[inner_slice] = p_unflat

        ############################ 8. Update boundary with the new values ############################################
        # Use the current existing context (is_nodal for all sides)
        self.boundary_manager.apply_boundary_on_single_var_all_sides(
            p2_full, self._nodal_bc_contexts
        )

        ############################ 9. Final Correction using Pressure Increment delta_p ##############################
        # Apply the update using the *solved increment*.
        # The buoyancy *is* adjusted now (updt_chi=1.0).
        CHI_UPDT_VALUE = 1.0
        self.pressure_solver.apply_pressure_gradient_update(
            p=p2_full,  # Use solved increment
            updt_chi=CHI_UPDT_VALUE,  # Adjust buoyancy now
            dt=dt,
            is_nongeostrophic=self.is_nongeostrophic,
            is_nonhydrostatic=self.is_nonhydrostatic,
        )

        ############################ 10. Update The Main Exner Pressure ################################################
        self.mpv.p2_nodes += p2_full

        ############################ 11. Apply Final Boundary Conditions ###############################################
        self.boundary_manager.apply_boundary_on_all_sides(self.variables.cell_vars)

        # Use the current existing contexts (is_nodal for all sides)
        self.boundary_manager.apply_boundary_on_single_var_all_sides(
            self.mpv.p2_nodes, self._nodal_bc_contexts
        )

    def _forward_momenta_update(
        self,
        cellvars: np.ndarray,
        dt: float,
        p2n: np.ndarray,
        dbuoy: np.ndarray,
        g: float,
    ):
        """Update the momenta

        Parameters
        ----------
        cellvars: np.ndarray
            The cell variables
        p2n: np.ndarray
            The nodal pressure variables
        dbuoy : np.ndarray
            The pressured perturbation of Chi: PX'
        g: float
            The gravity strength
        """
        coriolis = self.coriolis.strength
        adjusted_momenta = self.variables.adjust_background_wind(self.wind_speed, -1.0)

        # pressure gradient factor: (P/Gamma)
        rhoYovG = self.pressure_solver._calculate_P_over_Gamma(cellvars)

        # Calculate the Exner pressure perturbation (Pi^prime) gradiant (RHS of the momenta equations)
        dpdx, dpdy, dpdz = self.discrete_operator.gradient(p2n)

        ###############################################################################################################
        ## UPDATING VARIABLES
        ###############################################################################################################
        # Updates: First the shared terms without regarding which one is in the gravity direction
        # Horizontal momentum in x
        cellvars[..., VI.RHOU] -= dt * (
            rhoYovG * dpdx
            - coriolis[2] * adjusted_momenta[..., 1]
            + coriolis[1] * adjusted_momenta[..., 2]
        )
        if self.grid.ndim >= 2:
            cellvars[..., VI.RHOV] -= dt * (
                rhoYovG * dpdy
                - coriolis[0] * adjusted_momenta[..., 2]
                + coriolis[2] * adjusted_momenta[..., 0]
            )
        if self.grid.ndim == 3:
            cellvars[..., VI.RHOW] -= dt * (
                rhoYovG * dpdy
                - coriolis[1] * adjusted_momenta[..., 0]
                + coriolis[0] * adjusted_momenta[..., 1]
            )

        # Updates: The momentum in the direction of gravity
        # Find vertical vs horizontal velocities:
        cellvars[..., self.pressure_solver.vertical_momentum_index] -= dt * (
            (g / self.Msq) * dbuoy * self.is_nonhydrostatic
        )

    def _forward_pressure_update(
        self, cellvars: np.ndarray, dt: float, p2n: np.ndarray
    ):
        """Update the Exner pressure."""

        # Compute the weighting factor Y = (rhoY / rho) = Theta
        Y = cellvars[..., VI.RHOY] / cellvars[..., VI.RHO]

        # Compute the divergence of the pressure-weighted momenta: (Pu)_x + (Pv)_y + (Pw)_z where
        # P = rho*Y = rho*Theta
        pressure_weighted_momenta = self._calculate_enthalpy_weighted_momenta(
            self.variables.cell_vars
        )
        inner_slice = one_element_inner_slice(self.grid.ndim, full=False)
        self.mpv.rhs[inner_slice] = self.discrete_operator.divergence(
            pressure_weighted_momenta,
        )

        # Adjust wall boundary nodes (scale). Notice the side is set to be BdrySide.ALL.
        # This will apply the 'extra' method whenever the boundary is defined to be WALL.
        boundary_operation = [
            WallAdjustment(
                target_side=BdrySide.ALL, target_type=BdryType.WALL, factor=2.0
            )
        ]
        self.boundary_manager.apply_extra_all_sides(self.mpv.rhs, boundary_operation)

        # Calculate the derivative of the Exner pressure with respect to P
        dpidP = calculate_dpi_dp(cellvars[..., VI.RHOY], self.Msq)

        # Create node-to-cell index (slice(1, -1) in all directions)
        inner_idx = one_element_inner_slice(self.grid.ndim, full=False)

        # Create a nodal variable to store the intermediate updates
        dp2n = np.zeros_like(p2n)
        dp2n[inner_idx] -= dt * dpidP * self.mpv.rhs[inner_slice]
        self.mpv.p2_nodes[...] += self.is_compressible * dp2n

    def _forward_buoyancy_update(self, cellvars: np.ndarray, dt: float):
        """Update the X' variable (rho X in the variables)

        Parameters
        ----------
        cellvars: np.ndarray
           The cell variables
        dt : float
            The time step
        """
        # get Chi variable and the derivative
        S0c = self.mpv.get_S0c_on_cells()
        dSdy = self.mpv.compute_dS_on_nodes()

        # Intermediate variable for current Chi
        currentX = cellvars[..., VI.RHO] * (
            (cellvars[..., VI.RHO] / cellvars[..., VI.RHOY]) - S0c
        )

        ###############################################################################################################
        # Update the variable
        ###############################################################################################################
        cellvars[..., VI.RHOX] = (
            currentX
            - dt
            * cellvars[..., self.pressure_solver.vertical_momentum_index]
            * dSdy
            * cellvars[..., VI.RHO]
        )

    def _calculate_enthalpy_weighted_momenta(self, cellvars: np.ndarray):
        """Calculate the vector [Pu, [Pv], [Pw]] which is equal to rho*Theta*velocities. This is needed in multiple
        parts of the code"""
        Y = cellvars[..., VI.RHOY] / cellvars[..., VI.RHO]
        momenta_indices = [VI.RHOU, VI.RHOV, VI.RHOW][: self.grid.ndim]
        return cellvars[..., momenta_indices] * Y[..., np.newaxis]

    def get_dt(self):
        pass


def example_usage():
    from atmpy.physics.thermodynamics import Thermodynamics
    from atmpy.grid.utility import DimensionSpec, create_grid
    from atmpy.variables.variables import Variables
    from atmpy.infrastructure.enums import (
        VariableIndices as VI,
        HydrostateIndices as HI,
    )
    from atmpy.boundary_conditions.bc_extra_operations import (
        WallAdjustment,
        PeriodicAdjustment,
    )

    np.set_printoptions(linewidth=300)
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=10)

    ####################################################################################################################
    # GRID DATA ########################################################################################################
    ####################################################################################################################

    nx = 6
    ngx = 2
    nnx = nx + 2 * ngx
    ny = 10
    ngy = 2
    nny = ny + 2 * ngy

    dim = [DimensionSpec(nx, 0, 2, ngx), DimensionSpec(ny, 0, 2, ngy)]
    grid = create_grid(dim)

    th = Thermodynamics()

    ####################################################################################################################
    ######### MPV ######################################################################################################
    ####################################################################################################################
    from atmpy.variables.multiple_pressure_variables import MPV

    A0 = 0.1 / 100
    t_ref = 100.0
    T_ref = 300.0
    R_gas = 287.4
    h_ref = 10_000
    cp = th.gamma * R_gas / (th.gm1)
    N_ref = 9.81 / np.sqrt(cp * T_ref)

    grav = 9.81

    g = grav * h_ref / (R_gas * T_ref)
    Nsq_ref = N_ref * N_ref

    Msq = 0.115
    gravity_vec = [0.0, g, 0.0]

    mpv = MPV(grid)
    mpv.state(gravity_vec, Msq)
    Y_bar = mpv.hydrostate.cell_vars[..., HI.Y0]
    rhobar = mpv.hydrostate.cell_vars[..., HI.RHO0]

    ####################################################################################################################
    ########## Prepare some thermodynamic constant for initialization ##################################################
    ####################################################################################################################
    rhobar_n = mpv.hydrostate.node_vars[..., HI.RHOY0]
    Y_bar_n = mpv.hydrostate.node_vars[..., HI.Y0]
    oorhobarsqrt = 1.0 / np.sqrt(rhobar)
    oorhobarsqrt_n = 1.0 / np.sqrt(rhobar_n)

    Cs = np.sqrt(th.gamma / Msq)
    N = t_ref * np.sqrt(Nsq_ref)

    k = N / Cs

    omega = 7.292 * 1e-5
    coriolis = [0.0, 0.0, 0.0]
    coriolis[2] = 2.0 * omega * t_ref
    F = coriolis[2]

    G = np.sqrt(9.0 / 40.0)
    Gamma = G * N / Cs
    mu = -Gamma

    matrix = -np.array(
        [
            [0, F, 0, 1j * Cs * k],
            [-F, 0, -N, Cs * (mu + Gamma)],
            [0, N, 0, 0],
            [1j * Cs * k, Cs * (mu - Gamma), 0, 0],
        ]
    )

    eigval, eigvec = np.linalg.eig(matrix)
    ind = np.argmax(np.real(eigval))
    x = grid.x_cells.reshape(-1, 1)
    y = grid.y_cells.reshape(-1, 1)
    X, Y = np.meshgrid(x, y)
    t = 0
    s = 1
    exponentials = np.exp(1j * k * X + mu * Y + (eigval[ind]) * (t) + 1j * s * t)
    chi_Y = A0 * np.real(eigvec[2, ind] * exponentials).T
    Y_p = oorhobarsqrt * N / 9.81 * Y_bar * chi_Y

    Theta = Y_bar + Y_p

    chi_u = A0 * np.real(eigvec[0, ind] * exponentials).T
    chi_w = A0 * np.real(eigvec[1, ind] * exponentials).T

    up = oorhobarsqrt * chi_u
    vp = oorhobarsqrt * chi_w

    x = grid.x_nodes.reshape(-1, 1)
    y = grid.y_nodes.reshape(-1, 1)
    X, Y = np.meshgrid(x, y)
    t = 0
    s = 1
    exponentials = np.exp(1j * k * X + mu * Y + (eigval[ind]) * (t) + 1j * s * t)
    chi_pi = A0 * np.real(eigvec[3, ind] * exponentials).T

    pi_n = oorhobarsqrt_n * Cs / Y_bar_n / th.Gammainv * chi_pi

    ####################################################################################################################
    ## VARIABLE DATA ###################################################################################################
    ####################################################################################################################

    rng = np.random.default_rng()
    arr = np.arange(nnx * nny)
    rng.shuffle(arr)
    array = arr.reshape(nnx, nny)

    variables = Variables(grid, 6, 1)
    variables.cell_vars[..., VI.RHO] = rhobar
    # variables.cell_vars[..., VI.RHO][1:-1, 1:-1] = 4
    variables.cell_vars[..., VI.RHOU] = up
    variables.cell_vars[..., VI.RHOY] = rhobar * Theta
    variables.cell_vars[..., VI.RHOW] = 0.0

    rng.shuffle(arr)
    array = arr.reshape(nnx, nny)
    variables.cell_vars[..., VI.RHOV] = vp

    mpv.p2_nodes[...] = pi_n

    ####################################################################################################################
    ######### FLUX #####################################################################################################
    ####################################################################################################################
    from atmpy.physics.eos import ExnerBasedEOS
    from atmpy.flux.flux import Flux

    eos = ExnerBasedEOS()
    flux = Flux(grid, variables, eos)

    ########## STRATIFICATION ##########################################################################################
    def stratification_function(y):
        Nsq = Nsq_ref * t_ref * t_ref
        g1 = g / Msq

        return np.exp(Nsq * y / g1)

    stratification = stratification_function

    ####################################################################################################################
    ######### BOUNDARY MANAGER #########################################################################################
    ####################################################################################################################
    from atmpy.boundary_conditions.boundary_manager import BoundaryManager
    from atmpy.boundary_conditions.contexts import (
        BCInstantiationOptions,
        BoundaryConditionsConfiguration,
        BCApplicationContext,
    )

    direction = "y"
    bc = BCInstantiationOptions(
        side=BdrySide.BOTTOM,
        type=BdryType.REFLECTIVE_GRAVITY,
        direction=direction,
        grid=grid,
        stratification=stratification,
    )
    bc2 = BCInstantiationOptions(
        side=BdrySide.TOP,
        type=BdryType.REFLECTIVE_GRAVITY,
        direction=direction,
        grid=grid,
        stratification=stratification,
    )
    bc3 = BCInstantiationOptions(
        side=BdrySide.LEFT, type=BdryType.PERIODIC, direction="x", grid=grid
    )
    bc4 = BCInstantiationOptions(
        side=BdrySide.RIGHT, type=BdryType.PERIODIC, direction="x", grid=grid
    )
    options = [bc, bc2, bc3, bc4]
    # options = [bc, bc2]

    bc_conditions = BoundaryConditionsConfiguration(options)
    manager = BoundaryManager(bc_conditions)
    manager.apply_boundary_on_all_sides(variables.cell_vars)

    ####################################################################################################################
    ########## DISCRETE OPERATOR AND PRESSURE SOLVER ###################################################################
    ####################################################################################################################
    from atmpy.infrastructure.enums import (
        PressureSolvers,
        DiscreteOperators,
        LinearSolvers,
    )
    from atmpy.pressure_solver.contexts import (
        DiscreteOperatorsContext,
        PressureContext,
    )

    from atmpy.physics.gravity import Gravity
    from atmpy.time_integrators.coriolis import CoriolisOperator

    gravity = Gravity(gravity_vec, grid.ndim)
    coriolis = CoriolisOperator([0.0, 1.0, 0.0], gravity)

    op_context = DiscreteOperatorsContext(
        operator_type=DiscreteOperators.CLASSIC_OPERATOR, grid=grid
    )
    linear_solver = LinearSolvers.BICGSTAB

    # Instantiate the pressure solver context by specifying enums for pressure solver and discrete operator.
    ps_context: PressureContext[ClassicalPressureSolver] = PressureContext(
        solver_type=PressureSolvers.CLASSIC_PRESSURE_SOLVER,
        op_context=op_context,
        linear_solver_type=linear_solver,
        precondition_type=Preconditioners.DIAGONAL,
        extra_dependencies={
            "grid": grid,
            "variables": variables,
            "mpv": mpv,
            "boundary_manager": manager,
            "coriolis": coriolis,
            "Msq": Msq,
            "thermodynamics": th,
        },
    )

    pressure = ps_context.instantiate()

    ####################################################################################################################
    ######## TIME INTEGRATION ##########################################################################################
    ####################################################################################################################

    ##### Approach 1 ######################################################
    ### Using instantiation context for central instantiation of all classes:
    from atmpy.time_integrators.contexts import TimeIntegratorContext
    from atmpy.infrastructure.enums import TimeIntegrators

    dt = 0.01
    context: TimeIntegratorContext[IMEXTimeIntegrator] = TimeIntegratorContext(
        integrator_type=TimeIntegrators.IMEX,
        grid=grid,
        variables=variables,
        flux=flux,
        boundary_manager=manager,
        advection_routine=AdvectionRoutines.STRANG_SPLIT,
        dt=dt,
        extra_dependencies={
            "mpv": mpv,
            "pressure_solver": pressure,
            "wind_speed": [0.0, 0.0, 0.0],  # optional: override default wind speed
            "is_nonhydrostatic": True,
            "is_nongeostrophic": True,
            "is_compressible": True,
        },
    )
    time_integrator = context.instantiate()

    ###### Approach 2 #############################
    # Or simply using the direct needed integrator:
    # time_integrator = IMEXTimeIntegrator(
    #     grid=grid,
    #     variables=variables,
    #     mpv=mpv,
    #     flux=flux,
    #     boundary_manager=manager,
    #     coriolis_operator=coriolis,
    #     pressure_solver=pressure,
    #     thermodynamics=th,
    #     dt=0.1,
    #     Msq=1.0
    # )
    # print(variables.cell_vars[..., VI.RHOY] / variables.cell_vars[..., VI.RHO])
    print(variables.cell_vars[..., VI.RHOY])
    # manager.apply_boundary_on_all_sides(variables.cell_vars)
    # print(variables.cell_vars[..., VI.RHOU])

    # print(mpv.wcenter)
    # print(mpv.p2_nodes)
    # time_integrator.forward_update()
    # time_integrator.backward_explicit_update(dt)
    # # contexts = [BCApplicationContext(is_nodal=True)] * grid.ndim * 2
    # #
    # # # Update the boundary nodes for pressure variable
    # # manager.apply_boundary_on_single_var_all_sides(
    # #     mpv.p2_nodes, contexts
    # # )
    # # # print(mpv.wcenter)
    # print(mpv.p2_nodes)
    # print(variables.cell_vars[..., VI.RHOU])
    # print(".......................................................")
    # print(variables.cell_vars[..., VI.RHOV])
    # print(".......................................................")

    print(mpv.p2_nodes)
    # # pressure.pressure_coefficients_nodes(variables.cell_vars, dt)
    # time_integrator.forward_update()
    # time_integrator.backward_update_explicit(dt)
    time_integrator.backward_update_implicit(dt)
    # # x = pressure.helmholtz_operator(mpv.p2_nodes, dt, True, True, True)
    print(mpv.p2_nodes)
    # print(x)
    #
    # rhs = np.ones_like(x).flatten()
    # print(mpv.wcenter.shape)
    # y, info = pressure.solve_helmholtz(rhs, dt, True, True, True)
    # print(y.reshape((grid.ncx_total - 1, grid.ncy_total - 1)))
    # print(info)
    # print(variables.cell_vars[..., VI.RHOU])
    # manager.apply_boundary_on_all_sides(variables.cell_vars)
    # print(variables.cell_vars[..., VI.RHOU])


if __name__ == "__main__":
    example_usage()
