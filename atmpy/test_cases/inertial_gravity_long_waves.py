import numpy as np
from typing import TYPE_CHECKING
import matplotlib

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

if TYPE_CHECKING:
    from atmpy.variables.variables import Variables
    from atmpy.variables.multiple_pressure_variables import MPV
    from atmpy.physics.thermodynamics import Thermodynamics


class InertialGravityLongWaves(BaseTestCase):
    """
    Inertial Gravity Long Waves test case based on the setup described in PyBella's
    `test_internal_long_wave.py`.
    """

    def __init__(self, config_override: SimulationConfig = None):
        _effective_config: SimulationConfig
        run_setup_method = False

        if config_override is not None:
            _effective_config = config_override
        else:
            _effective_config = SimulationConfig()
            run_setup_method = True

        super().__init__(name="InertialGravityLongWaves", config=_effective_config)

        ########################### Test Case Specific Parameters ####################################################
        self.u0: float = 0.0  # Background velocity U
        self.v0: float = 0.0  # Background velocity V
        self.w0: float = 0.0  # Background velocity W

        # Match Project 1's scale_factor
        self.scale_factor = 1.0
        self.xc: float = 0.0

        if run_setup_method:
            self.setup()

    def setup(self):
        """Configure the SimulationConfig to match test_internal_long_wave.py."""
        print("Setting up Inertial Gravity Long Waves configuration...")

        #################################### Grid Configuration ########################################################
        nx = 301
        ny = 10

        grid_updates = {
            "ndim": 2,
            "nx": nx,
            "ny": ny,
            "nz": 0,
            "xmin": -15.0 * self.scale_factor,
            "xmax": 15.0 * self.scale_factor,
            "ymin": 0.0,
            "ymax": 1.0,
            "ngx": 2,
            "ngy": 2,
        }
        self.set_grid_configuration(grid_updates)

        #################################### Global Constants ##########################################################
        constants_updates = {
            "gamma": 1.4,
            "R_gas": 287.4,
            "p_ref": 1e5,
            "T_ref": 300.0,
            "h_ref": 10000.0,
            "t_ref": 100.0,
            "grav": 9.81,
            "Nsq_ref": 1.0e-4,
            "omega": 7.292e-5,
        }
        self.set_global_constants(constants_updates)

        #################################### Boundary Conditions #######################################################
        # Matches Project 1: PERIODIC in X, WALL in Y
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
        # Match Project 1 tout calculation
        tout_val = self.scale_factor * 1.0 * 3000.0 / self.config.global_constants.t_ref
        temporal_updates = {
            "CFL": 0.9,
            "dtfixed": 1.0,
            "dtfixed0": None,
            "tout": np.array([tout_val]),
            "tmax": tout_val + 1.0,  # Ensure it runs at least one tout
            "stepmax": 50000,
            "use_acoustic_cfl": True,
        }
        self.set_temporal(temporal_updates)

        #################################### Model Regimes #############################################################
        # Match Project 1: is_compressible = 1
        regime_updates = {
            "is_nongeostrophic": 1,
            "is_nonhydrostatic": 1,
            "is_compressible": 1,
        }
        self.set_model_regimes(regime_updates)

        #################################### Numerics ##################################################################
        # Match Project 1: initial_projection = False
        numerics_updates = {
            "limiter": LimiterType.AVERAGE,  # LimiterType.AVERAGE was default, NONE matches
            "riemann_solver": RiemannSolvers.MODIFIED_HLL,
            "reconstruction": FluxReconstructions.MODIFIED_MUSCL,
            "first_order_advection_routine": AdvectionRoutines.FIRST_ORDER_RK,
            "second_order_advection_routine": AdvectionRoutines.STRANG_SPLIT,
            "linear_solver": LinearSolvers.BICGSTAB,
            "preconditioner": Preconditioners.DIAGONAL,
            "initial_projection": False,  # Changed to match Project 1
        }
        self.set_numerics(numerics_updates)

        ################################### Physics Settings ###########################################################
        coriolis_strength = (
            self.config.global_constants.omega * self.config.global_constants.t_ref
        )
        physics_updates = {
            "wind_speed": [self.u0, self.v0, self.w0],
            "gravity_strength": (0.0, 1.0, 0.0),
            "coriolis_strength": (coriolis_strength, 0.0, 0.0),
            "stratification": self.inertial_gravity_waves_stratification,
        }
        self.set_physics(physics_updates)

        #################################### Outputs ###################################################################
        output_updates = {
            "output_type": "test",
            "output_folder": "inertial_gravity_long_wave",
            "output_base_name": "_inertial_long_wave",
            "output_timesteps": True,
            "output_frequency_steps": 10,
        }
        self.set_outputs(output_updates)
        self._update_output_suffix()  # Update suffix after grid change

        ################################################################################################################
        # Final check/update of Msq
        self._update_Msq()
        print(f"Configuration complete. Msq = {self.config.model_regimes.Msq}")
        print(
            f"Output files: {self.config.outputs.output_base_name}{self.config.outputs.output_suffix}"
        )

    def inertial_gravity_waves_stratification(self, y: float) -> float:
        """Analytical stratification profile, matching project 1."""
        gl = self.config.global_constants
        g_eff = self.config.physics.gravity_strength[1] / self.config.model_regimes.Msq
        Nsq_scaled = gl.Nsq_ref * (gl.t_ref**2)
        return np.exp(Nsq_scaled * y / g_eff)

    def molly_function(self, x: float):
        """Mollifying function, matching project 1."""
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
        Initialize fields based on a hydrostatically balanced state derived from a
        perturbed potential temperature field. This logic is a direct translation
        of the `sol_init` function from `test_internal_long_wave.py`.
        """
        print("Initializing solution for Inertial Gravity Waves (PyBella logic)...")

        grid = self.config.spatial_grid.grid
        gl = self.config.global_constants
        Msq = self.config.model_regimes.Msq

        # 1. Define perturbation parameters, matching Project 1
        delth = 0.01 / gl.T_ref
        a = self.scale_factor * 5.0e3 / gl.h_ref

        gravity = self.config.physics.gravity_strength
        mpv.state(gravity, Msq)

        # 2. Get full grid coordinates (cells and nodes)
        XC, YC = np.meshgrid(grid.x_cells, grid.y_cells, indexing="ij")
        XN, YN = np.meshgrid(grid.x_nodes[:-1], grid.y_nodes[:-1], indexing="ij")

        # 3. Calculate the total (background + perturbation) potential temperature field
        strat_func = self.config.physics.stratification
        Y_cells = strat_func(YC) + delth * self.molly_function(XC) * np.sin(
            np.pi * YC
        ) / (1.0 + (XC - self.xc) ** 2 / a**2)
        Y_nodes = strat_func(YN) + delth * self.molly_function(XN) * np.sin(
            np.pi * YN
        ) / (1.0 + (XN - self.xc) ** 2 / a**2)

        # 4. Use the translated 'column' function to compute the hydrostatically balanced state
        # for the given perturbed potential temperature profile.
        # Note: The function operates on 1D vertical profiles. We take a central column.
        x_center_idx = grid.nx // 2
        HySt = mpv.compute_hydrostate_from_profile(
            Y_cells, Y_nodes, self.config.physics.gravity_strength, Msq
        )
        print("Computed 1D hydrostatically balanced state from perturbed profile.")

        # 5. Get slices for the full inner domain (excluding ghost cells)
        inner_slice = grid.get_inner_slice()
        ngx, ngy = grid.ngx, grid.ngy

        # 6. Set the solution variables from the balanced state
        # The hydrostatic state is 1D, so we broadcast it to the full 2D domain.
        rhoY0_2d = np.tile(mpv.hydrostate.cell_vars[:, HI.RHOY0], (grid.ncx_total, 1))
        rho_final = HySt.rhoY0 / Y_cells

        variables.cell_vars[..., VI.RHO] = rho_final
        variables.cell_vars[..., VI.RHOU] = rho_final * self.u0
        variables.cell_vars[..., VI.RHOV] = rho_final * self.v0
        variables.cell_vars[..., VI.RHOW] = rho_final * self.w0
        variables.cell_vars[..., VI.RHOY] = HySt.rhoY0

        mpv.p2_cells = HySt.p20
        mpv.p2_nodes[:, ngy:-ngy] = HySt.p20_nodes[:, ngy:-ngy]

        # Set buoyancy tracer
        inner_slice = tuple([slice(None, None), slice(2, -1)])
        if VI.RHOX < variables.num_vars_cell:
            # Tile the 1D background S0 for this calculation
            S0_nd = mpv.get_S0c_on_cells()  # This method tiles correctly
            buoyancy_tracer = rho_final[inner_slice] * (
                1.0 / Y_cells[inner_slice] - S0_nd[inner_slice]
            )
            variables.cell_vars[inner_slice][..., VI.RHOX] = buoyancy_tracer

        # 8. Call the translated 'initial_pressure' function for final correction
        mpv.correct_initial_pressure(variables, Msq)
        print("Applied initial pressure correction.")

        print("Solution initialization complete.")
