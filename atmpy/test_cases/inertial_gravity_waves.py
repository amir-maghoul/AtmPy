import numpy as np
from dataclasses import field  # Removed dataclass, not needed here
from typing import List, Tuple, Dict, Any, TYPE_CHECKING
import matplotlib
from _pytest import stash
from setuptools.sandbox import run_setup

from atmpy.boundary_conditions.contexts import BCApplicationContext

# matplotlib.use("TkAgg")

import logging

from atmpy.infrastructure.utility import (
    one_element_inner_nodal_shape,
    one_element_inner_slice,
    directional_indices,
    directional_full_inner_slice,
)
from atmpy.test_cases.base_test_case import BaseTestCase
from atmpy.configuration.simulation_configuration import SimulationConfig
from atmpy.infrastructure.enums import (
    BoundaryConditions as BdryType,
    BoundarySide,
    AdvectionRoutines,
    RiemannSolvers,
    FluxReconstructions,
    LinearSolvers,
    Preconditioners,
)
from atmpy.infrastructure.enums import (
    VariableIndices as VI,
    HydrostateIndices as HI,
    SlopeLimiters as LimiterType,
)
from atmpy.physics.thermodynamics import Thermodynamics

if TYPE_CHECKING:
    from atmpy.variables.variables import Variables
    from atmpy.variables.multiple_pressure_variables import MPV


class InertialGravityWaves(BaseTestCase):
    """
    Traveling Vortex test case based on the setup described in PyBella.
    This involves an isentropic vortex embedded in a uniform flow on a doubly
    periodic domain with zero gravity.
    """

    def __init__(self, config_override: SimulationConfig = None):
        # Initialize with a default SimulationConfig, which will be modified in setup
        _effective_config: SimulationConfig
        run_setup_method = False

        if config_override is not None:
            _effective_config = config_override
        else:
            # No override, create a default config. BaseTestCase will get this,
            # and then setup() will populate it.
            _effective_config = SimulationConfig()
            run_setup_method = True

        super().__init__(name="TravelingVortex", config=_effective_config)

        ############################### Vortex Specific Parameters #####################################################
        self.correct_distribution = True

        self.u0: float = 0.2  # Background velocity U
        self.v0: float = 0.0  # Background velocity V
        self.w0: float = 0.0  # Background velocity W

        # planetary -> 160.0;  long-wave -> 20.0;  standard -> 1.0;
        self.scale_factor = 1.0

        self.xc: float = -5 * self.scale_factor

        ###########################3 Call setup to configure the simulation ############################################
        if run_setup_method:
            self.setup()

    def setup(self):
        """Configure the SimulationConfig for the Traveling Vortex case."""
        print("Setting up Traveling Vortex configuration...")

        #################################### Grid Configuration ########################################################
        nx = 300
        ny = 10

        grid_updates = {
            "ndim": 2,
            "nx": nx,
            "ny": ny,
            "nz": 0,
            "xmin": -15 * self.scale_factor,
            "xmax": 15 * self.scale_factor,
            "ymin": 0,
            "ymax": 1,
            "ngx": 2,  # Standard ghost cells
            "ngy": 2,
        }
        self.set_grid_configuration(grid_updates)

        #################################### Global Constants ##########################################################

        constants_updates = {
            "gamma": 1.4,
            "R_gas": 287.4,
            "p_ref": 86_100,
            "T_ref": 300.0,
            "h_ref": 10_000.0,
            "t_ref": 100.0,
            "grav": 9.81,
            "Nsq_ref": 1.0e-4,
        }
        self.set_global_constants(constants_updates)

        #################################### Boundary Conditions #######################################################
        self.set_boundary_condition(
            BoundarySide.LEFT, BdryType.PERIODIC, mpv_type=BdryType.PERIODIC
        )
        self.set_boundary_condition(
            BoundarySide.RIGHT, BdryType.PERIODIC, mpv_type=BdryType.PERIODIC
        )
        self.set_boundary_condition(
            BoundarySide.BOTTOM, BdryType.REFLECTIVE_GRAVITY, mpv_type=BdryType.WALL
        )
        self.set_boundary_condition(
            BoundarySide.TOP, BdryType.REFLECTIVE_GRAVITY, mpv_type=BdryType.WALL
        )

        #################################### Temporal Setting ##########################################################
        temporal_updates = {
            "CFL": 0.9,
            "dtfixed": 5.0 * (12.5 / 15.0) * 0.5 * self.scale_factor * 30.0 / 100,
            "dtfixed0": None,
            "tout": np.array([0.0, 10.0]),
            "tmax": 30 * self.scale_factor,
            "stepmax": 50_000,
            "use_acoustic_cfl": True,  # If True adds max_sound_speed to the speed, therefore smaller dt in dynamic
        }
        self.set_temporal(temporal_updates)

        #################################### Model Regimes #############################################################
        regime_updates = {
            "is_nongeostrophic": 1,
            "is_nonhydrostatic": 1,
            "is_compressible": 1,
        }
        self.set_model_regimes(regime_updates)  # This also updates Msq

        #################################### Numerics  #################################################################
        numerics_updates = {
            "limiter": LimiterType.AVERAGE,
            "riemann_solver": RiemannSolvers.MODIFIED_HLL,
            "reconstruction": FluxReconstructions.MODIFIED_MUSCL,
            "first_order_advection_routine": AdvectionRoutines.FIRST_ORDER_RK,
            "second_order_advection_routine": AdvectionRoutines.STRANG_SPLIT,
            "linear_solver": LinearSolvers.BICGSTAB,
            "preconditioner": Preconditioners.DIAGONAL,
            "initial_projection": True,
        }
        self.set_numerics(numerics_updates)

        ################################### Physics Settings ###########################################################
        physics_updates = {
            "wind_speed": [self.u0, self.v0, self.w0],
            "gravity_strength": (0.0, 1.0, 0.0),  # Zero gravity case
            "coriolis_strength": (0.0, 0.0, 0.0),
            "stratification": self.inertial_gravity_waves_stratification,
        }
        self.set_physics(physics_updates)

        #################################### Outputs ###################################################################
        output_updates = {
            "output_type": "test",
            "output_folder": "inertial_gravity_waves",
            "output_base_name": "_inertial_gravity_waves",
            "output_timesteps": True,
            "output_frequency_steps": 10,
            # output_suffix is updated automatically based on grid
        }
        self.set_outputs(output_updates)

        #################################### Diagnostics ###############################################################
        diag_updates = {
            "diag": True,
            "diag_current_run": "atmpy_inertial_gravity_waves",
        }
        self.set_diagnostics(diag_updates)

        ################################################################################################################
        # Final check/update of Msq after constants are set
        self._update_Msq()
        # Final check/update of output suffix
        self._update_output_suffix()

        print(f"Configuration complete. Msq = {self.config.model_regimes.Msq}")
        print(
            f"Output files: {self.config.outputs.output_base_name}{self.config.outputs.output_suffix}"
        )

    def inertial_gravity_waves_stratification(self, y: float) -> float:
        """
        Isothermal stratification for the Inertial Gravity Waves case.
        """
        gl = self.config.global_constants
        g = self.config.physics.gravity_strength[1] / self.config.model_regimes.Msq
        Nsq = gl.Nsq_ref * (gl.t_ref**2)
        return np.exp(Nsq * y / g)

    def molly_function(self, x: float):
        grid = self.config.spatial_grid
        xmin = grid.xmin
        xmax = grid.xmax
        del0 = 0.25
        L = xmax - xmin
        xi_l = np.minimum(1.0, (x - xmin) / (del0 * L))
        xi_r = np.minimum(1.0, (xmax - x) / (del0 * L))
        return 0.5 * np.minimum(1.0 - np.cos(np.pi * xi_l), 1.0 - np.cos(np.pi * xi_r))

    def initialize_solution(self, variables: "Variables", mpv: "MPV"):
        """
        Initialize fields for the Inertial Gravity Waves case by creating a
        fully consistent, hydrostatically balanced 2D initial state.

        This corrected version uses an explicit double loop for the hydrostatic
        integration to ensure the 2D nature of the pressure field is preserved.
        """
        print(
            "Initializing a hydrostatically balanced solution (Robust Integration)..."
        )

        # --- 1. Get configuration, grid, and thermodynamic parameters ---
        grid = self.config.spatial_grid.grid
        Msq = self.config.model_regimes.Msq
        thermo = Thermodynamics()
        gravity_vec = self.config.physics.gravity_strength
        g = gravity_vec[1]  # Assuming gravity is in the y-direction

        # Perturbation parameters from the paper's setup (non-dimensionalized)
        delth = 0.01 / self.config.global_constants.T_ref
        a = 5000.0 / self.config.global_constants.h_ref
        xc_nd = 100000.0 / self.config.global_constants.h_ref
        H_nd = 10000.0 / self.config.global_constants.h_ref

        # --- 2. Define the total 2D potential temperature (Theta) field ---
        XC, YC = np.meshgrid(grid.x_cells, grid.y_cells, indexing="ij")
        strat_func = self.config.physics.stratification
        Theta0_background_2d = strat_func(YC)

        # Using the exact perturbation form from Skamarock and Klemp (1994), Eq. 39
        perturbation = (
            delth * np.sin(np.pi * YC / H_nd) / (1.0 + ((XC - xc_nd) / a) ** 2)
        )
        Theta_total_2d = Theta0_background_2d + perturbation

        # --- 3. Calculate the new, balanced hydrostatic state from the 2D Theta field ---
        # --- 3. Calculate the new, balanced hydrostatic state using a Trapezoidal Rule integration ---
        S_perturbed_2d = 1.0 / Theta_total_2d
        rhoY0_ref = 1.0
        pi0_ref = rhoY0_ref**thermo.gm1

        # Create the 2D Exner pressure field we will populate
        pi_hydro_2d = np.zeros_like(XC)

        # Grid parameters for the loop
        y_cells = grid.y_cells
        dy = grid.dy
        nx, ny = grid.cshape
        ref_idx = np.argmin(np.abs(y_cells))  # Find index of y=0

        # Set the reference pressure along the entire horizontal line at y=0
        pi_hydro_2d[:, ref_idx] = pi0_ref

        # Constant for the integration
        integration_constant = 0.5 * thermo.Gamma * g * dy

        # Integrate upwards from the reference level using the Trapezoidal Rule
        # pi_j = pi_{j-1} - C * (S_j + S_{j-1})
        for i in range(nx):  # Loop over every horizontal position
            for j in range(ref_idx + 1, ny):  # Loop up the column
                # The pressure change depends on the average S between the two levels
                avg_S = S_perturbed_2d[i, j] + S_perturbed_2d[i, j - 1]
                pi_hydro_2d[i, j] = pi_hydro_2d[i, j - 1] - integration_constant * avg_S

        # Integrate downwards from the reference level using the Trapezoidal Rule
        # pi_j = pi_{j+1} + C * (S_j + S_{j+1})
        for i in range(nx):  # Loop over every horizontal position
            for j in range(ref_idx - 1, -1, -1):  # Loop down the column
                avg_S = S_perturbed_2d[i, j] + S_perturbed_2d[i, j + 1]
                pi_hydro_2d[i, j] = pi_hydro_2d[i, j + 1] + integration_constant * avg_S

        # --- Sanity Check: Add this diagnostic right after the loops ---
        horizontal_std_dev = np.std(pi_hydro_2d, axis=0)
        print(
            f"DEBUG: Maximum horizontal standard deviation of pi_hydro is: {np.max(horizontal_std_dev)}"
        )
        # This value MUST be significantly greater than zero. If it is, the integration worked.
        # -------------------------------------------------------------

        # Calculate the other balanced fields from the new 2D Exner pressure
        rhoY_hydro_2d = pi_hydro_2d**thermo.gm1inv
        rho_final_2d = rhoY_hydro_2d * S_perturbed_2d
        rhoY0_ref = 1.0
        pi0_ref = rhoY0_ref**thermo.gm1

        # Explicit and robust hydrostatic integration
        y_cells = grid.y_cells
        dy = grid.dy
        pi_hydro_2d = np.zeros_like(XC)
        ref_idx = np.argmin(np.abs(y_cells))  # Find index of y=0

        nx, ny = grid.cshape

        # Set the reference pressure along the entire horizontal line at y=0
        pi_hydro_2d[:, ref_idx] = pi0_ref

        # Integrate upwards from the reference level, column by column
        for i in range(nx):  # Loop over every horizontal position
            for j in range(ref_idx + 1, ny):  # Loop up the column
                # The pressure at (i, j) depends on the pressure at (i, j-1)
                pi_hydro_2d[i, j] = (
                    pi_hydro_2d[i, j - 1]
                    - thermo.Gamma * g * S_perturbed_2d[i, j - 1] * dy
                )

        # Integrate downwards from the reference level, column by column
        for i in range(nx):  # Loop over every horizontal position
            for j in range(ref_idx - 1, -1, -1):  # Loop down the column
                # The pressure at (i, j) depends on the pressure at (i, j+1)
                pi_hydro_2d[i, j] = (
                    pi_hydro_2d[i, j + 1]
                    + thermo.Gamma * g * S_perturbed_2d[i, j + 1] * dy
                )

        # Calculate the other balanced fields from the new 2D Exner pressure
        rhoY_hydro_2d = pi_hydro_2d**thermo.gm1inv
        rho_final_2d = rhoY_hydro_2d * S_perturbed_2d

        # --- 4. Set the final solution and pressure variables ---
        inner_slice = tuple(
            slice(grid.ng[i][0], -grid.ng[i][1] or None) for i in range(grid.ndim)
        )

        variables.cell_vars[inner_slice + (VI.RHO,)] = rho_final_2d[inner_slice]
        variables.cell_vars[inner_slice + (VI.RHOU,)] = (
            rho_final_2d[inner_slice] * self.u0
        )
        variables.cell_vars[inner_slice + (VI.RHOV,)] = (
            rho_final_2d[inner_slice] * self.v0
        )
        variables.cell_vars[inner_slice + (VI.RHOY,)] = rhoY_hydro_2d[inner_slice]

        if Msq > 1e-12:
            mpv.p2_cells[...] = pi_hydro_2d / Msq
            mpv.p2_nodes[1:-1, 1:-1] = 0.25 * (
                mpv.p2_cells[:-1, :-1]
                + mpv.p2_cells[1:, :-1]
                + mpv.p2_cells[:-1, 1:]
                + mpv.p2_cells[1:, 1:]
            )
        else:
            mpv.p2_cells[...] = 0.0
            mpv.p2_nodes[...] = 0.0

        print(
            "Solution initialization complete. The initial state is now hydrostatically balanced."
        )

    # def initialize_solution(self, variables: "Variables", mpv: "MPV"):
    #     """Initialize fields based on a hydrostatic state with a potential temperature perturbation."""
    #     print("Initializing solution for Inertial Gravity Waves...")
    #
    #     grid = self.config.spatial_grid.grid
    #     Msq = self.config.model_regimes.Msq
    #     delth = 0.01 / self.config.global_constants.T_ref
    #     # a = self.config.spatial_grid.xmax / 60
    #     a = 0.5
    #
    #     # 1. Ensure the hydrostatic base state is calculated
    #     gravity = self.config.physics.gravity_strength
    #     g_axis = self.config.physics.gravity.axis
    #     mpv.state(gravity, Msq)
    #
    #     # 2. Get slices and coordinates for the inner domain
    #
    #     ngy = self.config.spatial_grid.grid.ng[1][
    #         0
    #     ]
    #
    #     # Get slices for inner domain (excluding ghost cells)
    #     inner_slice = directional_full_inner_slice(grid.ndim, g_axis, ngy)
    #     XC, YC = np.meshgrid(
    #         grid.x_cells[inner_slice[0]],
    #         grid.y_cells[inner_slice[1]],
    #         indexing="ij",
    #     )
    #
    #     # Calculate the perturbation field
    #     strat_func = self.config.physics.stratification
    #     stratification_field = strat_func(YC)
    #     molly_field = self.molly_function(XC)
    #
    #     perturbation = delth * molly_field * np.sin(np.pi * YC) / (1.0 + ((XC - self.xc) ** 2 / a ** 2))
    #     perturbed_stratification = stratification_field + perturbation
    #     perturbed_stratification[np.isclose(perturbed_stratification, 0)] = 1.0
    #
    #     # 3. Get the hydrostatic background fields (constant in x)
    #     # We need to broadcast the 1D hydrostate arrays to the 2D inner domain shape
    #     rhoY0_cells = mpv.hydrostate.cell_vars[inner_slice[1], HI.RHOY0]
    #     rhoY0_2d = np.broadcast_to(rhoY0_cells[np.newaxis, :], XC.shape)
    #     rho_final = rhoY0_2d / perturbed_stratification
    #
    #     # 5. Set the solution variables in the inner domain
    #     # Main variables
    #     variables.cell_vars[inner_slice + (VI.RHO,)] = rho_final
    #     variables.cell_vars[inner_slice + (VI.RHOU,)] = rho_final * self.u0
    #     variables.cell_vars[inner_slice + (VI.RHOV,)] = rho_final * self.v0
    #     variables.cell_vars[inner_slice + (VI.RHOW,)] = rho_final * self.w0
    #     variables.cell_vars[inner_slice + (VI.RHOY,)] = rhoY0_2d
    #
    #     # Tracer for buoyancy (if used)
    #     if VI.RHOX < variables.num_vars_cell:
    #         S0_2d = np.broadcast_to(mpv.hydrostate.cell_vars[inner_slice[1], HI.S0], XC.shape)
    #         buoyancy_tracer = rho_final * (rho_final / rhoY0_2d - S0_2d)
    #         variables.cell_vars[inner_slice + (VI.RHOX,)] = buoyancy_tracer
    #
    #
    #     # Nodal perturbation (initialize to hydrostatic state, i.e., zero perturbation)
    #     p0_nodes = mpv.hydrostate.node_vars[..., HI.P0]
    #     rhoY0_nodes = mpv.hydrostate.node_vars[..., HI.RHOY0]
    #
    #     if Msq > 1e-12:
    #         p2_nodes_1d = (p0_nodes / rhoY0_nodes) / Msq
    #         # Broadcast the 1D nodal values to the full 2D nodal grid
    #         mpv.p2_nodes[...] = np.broadcast_to(
    #             p2_nodes_1d[np.newaxis, :],
    #             mpv.p2_nodes.shape
    #         )
    #     else:
    #         mpv.p2_nodes[...] = 0.0
    #
    #
    #     rho_total = variables.cell_vars[inner_slice + (VI.RHO,)]
    #     rhoY_total = variables.cell_vars[inner_slice + (VI.RHOY,)]  # This is rho * Theta
    #
    #     # Calculate the total potential temperature field
    #     # Theta = (rho * Theta) / rho
    #     Theta_total = rhoY_total / rho_total
    #
    #     # Get the background potential temperature from the hydrostate
    #     Y0_pre = mpv.hydrostate.cell_vars[inner_slice[1], HI.Y0]
    #     Y0_pre_2D = np.broadcast_to(Y0_pre[np.newaxis, :], XC.shape)
    #     Y0_cells = stratification_field
    #     Theta0_background = Y0_cells
    #
    #     # Finally, calculate the perturbation field theta_prime
    #     theta_prime = Theta_total - Theta0_background
    #
    #     T_ref = self.config.global_constants.T_ref
    #     print(f"Max of calculated theta_prime: {np.max(theta_prime):.4e} K")
    #     print(f"Min of calculated theta_prime: {np.min(theta_prime):.4e} K")
    #
    #     print("Solution initialization complete.")
