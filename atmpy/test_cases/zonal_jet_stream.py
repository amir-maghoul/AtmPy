import numpy as np
from scipy.integrate import cumulative_trapezoid
from typing import TYPE_CHECKING
import logging

from atmpy.test_cases.base_test_case import BaseTestCase
from atmpy.configuration.simulation_configuration import SimulationConfig
from atmpy.infrastructure.enums import (
    BoundaryConditions as BdryType,
    BoundarySide,
    RiemannSolvers,
    FluxReconstructions,
    SlopeLimiters as LimiterType,
    AdvectionRoutines,
    LinearSolvers,
    Preconditioners
)
from atmpy.infrastructure.enums import VariableIndices as VI, HydrostateIndices as HI
from atmpy.physics.thermodynamics import Thermodynamics

if TYPE_CHECKING:
    from atmpy.variables.variables import Variables
    from atmpy.variables.multiple_pressure_variables import MPV


def isothermal_stratification(z: float) -> float:
    """A simple isothermal stratification, returning a constant value."""
    return 1.0

def stable_stratification_for_qg(z: float) -> float:
    """
    Defines a constant stable stratification (N^2 > 0).
    The value is typically chosen to give a realistic Burger number.
    """
    Nsq_ref = 1.0e-4 # A standard value for the mid-latitude troposphere
    return Nsq_ref


class ZonalJetStream(BaseTestCase):
    """
    A test case for a geostrophically balanced zonal jet stream on a doubly periodic domain (xz)
    with wall boundaries in y. The jet flows in the x-direction and varies in the z-direction (u = u(z)).
    This flow is balanced by a pressure gradient in the z-direction (∂p/∂z).
    """

    def __init__(self, config_override: SimulationConfig = None):
        _effective_config: SimulationConfig
        run_setup_method = False

        if config_override is not None:
            _effective_config = config_override
        else:
            _effective_config = SimulationConfig()
            run_setup_method = True

        super().__init__(name="ZonalJetStream", config=_effective_config)

        ############################### Jet Specific Parameters ########################################################
        self.jet_U_max: float = 20.0  # Maximum velocity of the jet in m/s (dimensional)
        self.jet_width: float = 2.0  # Characteristic width of the jet
        self.jet_center_z: float = 0.0  # The z-location of the jet core

        ############################### Physical & Domain Parameters ###################################################
        self.rho0: float = 1.0  # Background density (dimensionless)
        self.p0: float = 1.0  # Background pressure (dimensionless)
        self.size = 20
        self.boxsize = 0.5 * self.size
        self.tmax = 5.0

        self.stratification = False

        self.dtfixed0 = None
        self.dtfixed = self.dtfixed0 if self.dtfixed0 else 0.001
        self.acoustic_cfl = True
        self.output_freq = 1

        # --- Coriolis Force ---
        # A constant Coriolis force is required for geostrophic balance.
        self.g = 0.0
        coriolis_factor = 1.0
        self.t_ref = 100.0
        force = 1.0e-4 * self.t_ref * coriolis_factor
        self.cor_f = [0.0, 0.0, 0.0]
        self.coriolis_axis = 1  # Assuming y is the vertical axis for rotation
        self.cor_f[self.coriolis_axis] = force

        # --- Model Regimes ---
        # For a pure geostrophic test, nongeostrophic should be 0.
        self.is_nongeostrophic = 1
        self.is_nonhydrostatic = 1
        self.is_compressible = 1

        if run_setup_method:
            self.setup()

    def setup(self):
        """Configure the SimulationConfig for the Zonal Jet Stream case."""
        print("Setting up Zonal Jet Stream configuration...")

        #################################### Grid Configuration ########################################################
        grid_updates = {
            "ndim": 3, "nx": 40, "ny": 3, "nz": 40,
            "xmin": -self.boxsize, "xmax": self.boxsize,
            "ymin": -self.boxsize, "ymax": self.boxsize,
            "zmin": -self.boxsize, "zmax": self.boxsize,
            "ngx": 2, "ngy": 2, "ngz": 2,
        }
        self.set_grid_configuration(grid_updates)

        #################################### Global Constants ##########################################################
        constants_updates = {
            "gamma": 1.4, "R_gas": 287.4, "p_ref": 1.0e5, "T_ref": 300.0,
            "t_ref": self.t_ref,
        }
        self.set_global_constants(constants_updates)

        #################################### Boundary Conditions #######################################################
        # Periodic in X and Z (direction of jet and its variation)
        # Wall in Y (vertical)
        self.set_boundary_condition(
            BoundarySide.LEFT, BdryType.PERIODIC, mpv_type=BdryType.PERIODIC
        )
        self.set_boundary_condition(
            BoundarySide.RIGHT, BdryType.PERIODIC, mpv_type=BdryType.PERIODIC
        )
        self.set_boundary_condition(
            BoundarySide.BOTTOM, BdryType.WALL, mpv_type=BdryType.WALL
        )
        self.set_boundary_condition(
            BoundarySide.TOP, BdryType.WALL, mpv_type=BdryType.WALL
        )
        self.set_boundary_condition(
            BoundarySide.FRONT, BdryType.WALL, mpv_type=BdryType.WALL
        )
        self.set_boundary_condition(
            BoundarySide.BACK, BdryType.WALL, mpv_type=BdryType.WALL
        )

        #################################### Temporal Setting ##########################################################
        temporal_updates = {
            "CFL": 0.8,
            "dtfixed": self.dtfixed,
            "dtfixed0": self.dtfixed0,
            "tout": np.array([0.0, self.tmax]),
            "tmax": self.tmax,
            "stepmax": 100_000,
            "use_acoustic_cfl": self.acoustic_cfl,
        }
        self.set_temporal(temporal_updates)

        #################################### Model Regimes #############################################################
        regime_updates = {
            "is_nongeostrophic": self.is_nongeostrophic,
            "is_nonhydrostatic": self.is_nonhydrostatic,
            "is_compressible": self.is_compressible,
        }
        self.set_model_regimes(regime_updates)

        #################################### Numerics ##################################################################
        numerics_updates = {
            "limiter": LimiterType.AVERAGE,
            "riemann_solver": RiemannSolvers.MODIFIED_HLL,
            "reconstruction": FluxReconstructions.MODIFIED_MUSCL,
            "first_order_advection_routine": AdvectionRoutines.FIRST_ORDER_RK,
            "second_order_advection_routine": AdvectionRoutines.STRANG_SPLIT,
            "linear_solver": LinearSolvers.BICGSTAB,
            "preconditioner": Preconditioners.DIAGONAL,
            "initial_projection": False,
        }
        self.set_numerics(numerics_updates)

        ################################### Physics Settings ###########################################################
        physics_updates = {
            "wind_speed": [0.0, 0.0, 0.0],  # Background wind is zero; defined by the jet.
            "gravity_strength": (0.0, self.g, 0.0),  # No gravity in this idealized case
            "coriolis_strength": self.cor_f,
            "stratification": isothermal_stratification if not self.stratification else stable_stratification_for_qg,
        }
        self.set_physics(physics_updates)

        #################################### Outputs ###################################################################
        output_updates = {
            "output_type": "test",
            "output_folder": "zonal_jet_stream",
            "output_base_name": "_zonal_jet_stream",
            "output_frequency_steps": 10,
        }
        self.set_outputs(output_updates)

        ################################################################################################################
        self._update_Msq()
        self._update_output_suffix()
        print(f"Configuration complete for Zonal Jet Stream. Msq = {self.config.model_regimes.Msq}")

    def initialize_solution(self, variables: "Variables", mpv: "MPV"):
        """
        Initializes a geostrophically balanced zonal jet u(z).
        The balance equation ∂p/∂z = ρ₀ * f * u(z) is solved numerically to find the
        corresponding pressure field p(z).
        """
        print("Initializing solution for 3D Zonal Jet Stream...")

        grid = self.config.spatial_grid.grid
        thermo = Thermodynamics()
        thermo.update(self.config.global_constants.gamma)
        Msq = self.config.model_regimes.Msq
        coriolis = self.config.physics.coriolis_strength[self.coriolis_axis]

        inner_slice = grid.get_inner_slice()

        # --- Get Cell-Centered Coordinates ---
        ZC = np.meshgrid(
            grid.x_cells[inner_slice[0]],
            grid.y_cells[inner_slice[1]],
            grid.z_cells[inner_slice[2]],
            indexing="ij",
        )[2]  # We only need the Z coordinates for this setup

        # --- 1. Define the Jet Velocity Profile u(z) ---
        u_jet_profile = self.jet_U_max * np.exp(
            -(((ZC - self.jet_center_z) / self.jet_width) ** 2)
        )

        # --- 2. Calculate the Geostrophically Balanced Pressure p(z) ---
        z_cells_1d = grid.z_cells[inner_slice[2]]
        u_jet_1d = self.jet_U_max * np.exp(
            -(((z_cells_1d - self.jet_center_z) / self.jet_width) ** 2)
        )
        # The integrand for geostrophic balance: dp/dz = rho * f * u
        integrand = self.rho0 * coriolis * u_jet_1d

        # Integrate numerically to find the pressure perturbation
        p_jet_perturbation_1d = cumulative_trapezoid(integrand, z_cells_1d, initial=0)
        p_jet_base_1d = self.p0 + Msq * p_jet_perturbation_1d

        # Broadcast the 1D pressure array p(z) to the full 3D grid
        p_total_cells = p_jet_base_1d.reshape(1, 1, -1)

        # --- 3. Set Total State Variables ---
        rho_total = np.full_like(u_jet_profile, self.rho0)
        u_total = u_jet_profile
        v_total = np.zeros_like(u_total)
        w_total = np.zeros_like(u_total)

        # Calculate potential temperature field
        if self.config.model_regimes.is_compressible:
            p_total_safe = np.maximum(p_total_cells, 1e-9)
            rhoY_total = p_total_safe ** thermo.gamminv
        else:
            rhoY_total = np.ones_like(rho_total)  # Isentropic for incompressible case

        # --- 4. Assign to Cell Variables (Inner Domain Only) ---
        variables.cell_vars[inner_slice + (VI.RHO,)] = rho_total
        variables.cell_vars[inner_slice + (VI.RHOU,)] = rho_total * u_total
        variables.cell_vars[inner_slice + (VI.RHOV,)] = rho_total * v_total
        variables.cell_vars[inner_slice + (VI.RHOW,)] = rho_total * w_total
        variables.cell_vars[inner_slice + (VI.RHOY,)] = rhoY_total

        # --- 5. Handle Nodal Pressure Perturbation (p2) ---
        # The solver uses p2, the perturbation from the hydrostatic state.
        # Here, the entire geostrophic pressure gradient is the perturbation.
        z_nodes_1d = grid.z_nodes[inner_slice[2]]
        u_jet_nodes_1d = self.jet_U_max * np.exp(
            -(((z_nodes_1d - self.jet_center_z) / self.jet_width) ** 2)
        )
        integrand_nodes = self.rho0 * coriolis * u_jet_nodes_1d
        p_pert_nodes_1d = cumulative_trapezoid(integrand_nodes, z_nodes_1d, initial=0)

        # The model solves for the dynamic pressure p2. We set it directly.
        p2_nodes_unscaled = p_pert_nodes_1d.reshape(1, 1, -1)

        # We need rhoY at the nodes to convert the pressure perturbation to the solver's variable
        p_total_nodes = self.p0 + Msq * p2_nodes_unscaled
        rhoY_nodes = p_total_nodes ** thermo.gamminv

        mpv.p2_nodes[inner_slice] = thermo.Gamma * np.divide(
            p2_nodes_unscaled, rhoY_nodes, where=rhoY_nodes != 0
        )

        logging.info("3D Zonal Jet Stream initialization complete.")