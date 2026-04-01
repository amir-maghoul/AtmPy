

import numpy as np
from dataclasses import field
from typing import List, Tuple, Dict, Any, TYPE_CHECKING
import logging

from sphinx.testing.fixtures import if_graphviz_found
from atmpy.variables.utility import cell_averaging, cells_to_nodes_averaging
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


class BaroclinicQGVortex(BaseTestCase):

    def __init__(self, config_override: SimulationConfig = None):
        _effective_config: SimulationConfig
        run_setup_method = False

        if config_override is not None:
            _effective_config = config_override
        else:
            _effective_config = SimulationConfig()
            run_setup_method = True

        super().__init__(name="BaroclinicQGVortex", config=_effective_config)

        ############################### Vortex Specific Parameters #####################################################
        self.u0: float = 0.5  # Background velocity U (x)
        self.v0: float = 0.0  # Background velocity V (y - vertical)
        self.w0: float = 0.0  # Background velocity W (z)
        self.p0: float = 1.0  # Background pressure

        # Vortex Geometry
        self.xc: float = 0.0
        self.yc: float = 0.0
        self.zc: float = 0.0

        # Scales
        self.size = 1.0
        self.boxsize = 1.0 * self.size
        self.h_scale = 1.0
        self.y_boxscale = 0.1  # Height of domain relative to width
        self.R0: float = 0.2 * self.size  # Horizontal radius
        self.H_frac: float = 0.4 # Vertical aspect ratio of the vortex bubble

        # Amplitude (Exner pressure perturbation amplitude)
        # Needs to be small for QG validity (Rossby << 1)
        self.p_amp: float = -1.0e-5

        # Simulation settings
        self.tmax = 2.0 * self.size
        self.stepmax = 100000
        self.output_freq = 1
        self.checkpoint_freq = 1
        self.dtfixed0 = None
        self.dtfixed = 0.001  # Conservative time step

        # Physics
        self.baroclinic = True
        self.stratified_atmosphere = True
        self.h_ref = 10_000.0
        self.t_ref = 100.0
        factor = 200
        self.Nsq_ref = 1.0e-4*0.1

        self.g = 1.0
        self.coriolis_strength =  1.0e-4 * self.t_ref * factor # f parameter
        self.rossby_adjustment = 0.0
        self.sponge_layer = False
        self.sponge_width = 6
        self.sponge_axis = [0, 1, 2]
        self.sponge_amplitude = 0.5
        self.sponge_power = 2

        # Switches for testing
        QG = True
        self.initial_projection = True
        self.is_nongeostrophic = 1
        self.is_nonhydrostatic =  1
        self.is_compressible = 1

        if QG:
            self.initial_projection = False
            self.is_nongeostrophic = 0
            self.is_nonhydrostatic = 0
            self.is_compressible = 0


        if run_setup_method:
            self.setup()

    def setup(self):
        """Configure the SimulationConfig for the 3D Traveling Vortex case."""
        print("Setting up 3D Traveling Vortex QG configuration...")

        #################################### Grid Configuration ########################################################
        nx = nz = 40
        ny = 10

        # Domain extent
        Lx = self.boxsize * self.h_scale
        Ly = self.boxsize * self.y_boxscale

        grid_updates = {
            "ndim": 3,
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "xmin": -Lx,
            "xmax": Lx,
            "ymin": -Ly,
            "ymax": Ly,
            "zmin": -Lx,
            "zmax": Lx,
            "ngx": 2,
            "ngy": 2,
            "ngz": 2,
        }
        self.set_grid_configuration(grid_updates)

        #################################### Global Constants ##########################################################
        constants_updates = {
            "gamma": 1.4,
            "R_gas": 287.4,
            "p_ref": 1.0e5,
            "T_ref": 300.0,
            "h_ref": self.h_ref,
            "t_ref": self.t_ref,
            "Nsq_ref": self.Nsq_ref,
            "grav": 10.0,
        }
        self.set_global_constants(constants_updates)

        #################################### Boundary Conditions #######################################################
        # Periodic X/Z, Wall Y
        self.set_boundary_condition(BoundarySide.LEFT, BdryType.PERIODIC, mpv_type=BdryType.PERIODIC)
        self.set_boundary_condition(BoundarySide.RIGHT, BdryType.PERIODIC, mpv_type=BdryType.PERIODIC)
        self.set_boundary_condition(BoundarySide.BOTTOM, BdryType.REFLECTIVE_GRAVITY, mpv_type=BdryType.WALL)
        self.set_boundary_condition(BoundarySide.TOP, BdryType.REFLECTIVE_GRAVITY, mpv_type=BdryType.WALL)
        self.set_boundary_condition(BoundarySide.FRONT, BdryType.PERIODIC, mpv_type=BdryType.PERIODIC)
        self.set_boundary_condition(BoundarySide.BACK, BdryType.PERIODIC, mpv_type=BdryType.PERIODIC)

        self.config.boundary_conditions.sponge.active = self.sponge_layer
        self.config.boundary_conditions.sponge.width = self.sponge_width
        self.config.boundary_conditions.sponge.power = self.sponge_power
        self.config.boundary_conditions.sponge.amplitude = self.sponge_amplitude
        self.config.boundary_conditions.sponge.target_vars = ["RHOU", "RHOV", "RHOW"]
        self.config.boundary_conditions.sponge.axis = self.sponge_axis

        #################################### Temporal Setting ##########################################################
        temporal_updates = {
            "CFL": 0.9,
            "dtfixed": self.dtfixed,
            "dtfixed0": self.dtfixed0,
            "tout": np.array([0.0, self.tmax]),
            "tmax": self.tmax,
            "stepmax": self.stepmax,
            "use_acoustic_cfl": True,
        }
        self.set_temporal(temporal_updates)

        #################################### Model Regimes #############################################################
        regime_updates = {
            "is_nongeostrophic": self.is_nongeostrophic,
            "is_nonhydrostatic": self.is_nonhydrostatic,
            "is_compressible": self.is_compressible,
            # "Msq": self.Msq,
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
            "initial_projection": self.initial_projection,  # Projects initial velocity to be divergence free if needed
        }
        self.set_numerics(numerics_updates)

        ################################### Physics Settings ###########################################################
        # Gravity is along Y (axis 1)
        g = self.g
        cor = [0.0, self.coriolis_strength, 0.0]

        physics_updates = {
            "wind_speed": [self.u0, self.v0, self.w0],
            "gravity_strength": (0.0, g, 0.0),
            "coriolis_strength": cor,
            "stratification": self.qg_stratification,  # Analytical stratification
        }
        self.set_physics(physics_updates)

        #################################### Outputs ###################################################################
        output_updates = {
            "output_type": "test",
            "output_folder": "baroclinic_qg_vortex",
            "output_base_name": "_baroclinic_qg_vortex",
            "checkpoint_base_name": "_baroclinic_qg_vortex_checkpoint",
            "output_timesteps": True,
            "checkpoint_frequency_steps": self.checkpoint_freq,
            "output_frequency_steps": self.output_freq,
        }
        self.set_outputs(output_updates)

        #################################### Diagnostics ###############################################################
        diag_updates = {
            "diag": True,
            "diag_current_run": "atmpy_baroclinic_qg_vortex",
        }
        self.set_diagnostics(diag_updates)

        # if not self.Msq:
        self._update_Msq()
        self._update_output_suffix()
        print(f"Config done. Msq={self.config.model_regimes.Msq}, f={self.coriolis_strength}")

    def qg_stratification(self, y: np.ndarray):
        """Analytical stratification profile (N^2 = constant)."""
        gl = self.config.global_constants
        g_eff = self.config.physics.gravity_strength[1] / self.config.model_regimes.Msq
        # Exponent for constant N^2: Y ~ exp(N^2 * y / g)
        # This ensures dChi = d(1/Y)/dy is non-zero, preventing singularity in hydrostatic mode.
        exponent = (self.Nsq_ref * gl.t_ref ** 2) * y / g_eff
        return np.exp(exponent)

    def _get_coordinates_3d(self, grid, inner_slice, nodal=False):
        """Helper to get 3D coordinate meshgrids."""
        if nodal:
            x = grid.x_nodes[inner_slice[0]]
            y = grid.y_nodes[inner_slice[1]]
            z = grid.z_nodes[inner_slice[2]]
        else:
            x = grid.x_cells[inner_slice[0]]
            y = grid.y_cells[inner_slice[1]]
            z = grid.z_cells[inner_slice[2]]

        return np.meshgrid(x, y, z, indexing="ij")

    def _gaussian_perturbation(self, X, Y, Z, H_scale):
        """
        Calculates a perturbation that is Gaussian in Horizontal,
        but Compact Polynomial in Vertical.
        """
        # 1. Grid setup
        Lx = self.config.spatial_grid.grid.x_end - self.config.spatial_grid.grid.x_start
        Lz = self.config.spatial_grid.grid.z_end - self.config.spatial_grid.grid.z_start

        # Define a vertical cutoff height slightly smaller than the domain half-height
        # This ensures it hits zero before touching the boundary ghost cells.
        Ly_domain = self.config.spatial_grid.grid.y_end
        H_cutoff = H_scale  # Use 95% of the domain height

        # 2. Periodicity handling (X and Z only)
        dx = X - self.xc
        dz = Z - self.zc
        dy = Y - self.yc  # No periodicity check needed for Y (wall)

        dx = dx - Lx * np.round(dx / Lx)
        dz = dz - Lz * np.round(dz / Lz)

        # 3. Horizontal Part (Standard Gaussian)
        r2_h = dx ** 2 + dz ** 2
        # Note: Horizontal derivative logic remains the same
        horiz_profile = np.exp(- (r2_h / (2 * self.R0 ** 2)))

        d_horiz_dx = horiz_profile * (-dx / self.R0 ** 2)
        d_horiz_dz = horiz_profile * (-dz / self.R0 ** 2)

        # 4. Vertical Part (Compact Polynomial)
        # Define mask for where we are inside the bubble
        is_inside = np.abs(dy) < H_cutoff

        # Normalize y to range [0, 1]
        eta = dy / H_cutoff

        # Initialize vertical arrays (default to 0.0 outside)
        vert_profile = np.zeros_like(dy)
        d_vert_dy = np.zeros_like(dy)

        # Calculate profile only inside the cutoff
        # Formula: (1 - eta^2)^3
        # This looks very similar to a Gaussian but hits 0 exactly at eta=1
        term = (1 - eta[is_inside] ** 2)

        vert_profile[is_inside] = term ** 3
        # term = np.exp(- (eta**2))
        # vert_profile[...] = term


        # Derivative chain rule: 3 * (1-eta^2)^2 * (-2*eta) * (1/H_cutoff)
        d_vert_dy[is_inside] = 3 * (term ** 2) * (-2 * eta[is_inside]) * (1.0 / H_cutoff)
        # d_vert_dy[...] = (2*(dy/H_cutoff) **2) * term


        # 5. Combine
        val = self.p_amp * horiz_profile * vert_profile

        # Product Rule Derivatives
        d_val_dx = self.p_amp * d_horiz_dx * vert_profile
        d_val_dz = self.p_amp * d_horiz_dz * vert_profile
        d_val_dy = self.p_amp * horiz_profile * d_vert_dy

        return val, d_val_dx, d_val_dy, d_val_dz

    def _reshape_hydrostate_into_3D(self,variables: "Variables", mpv: "MPV", variable_1d:np.ndarray, nodal:bool = False):
        """ Reshape the given hydrostate 1D profile into 3D by broadcasting. """
        g_axis = self.config.physics.gravity.axis
        grid = self.config.spatial_grid.grid
        ng_g_axis = self.config.spatial_grid.grid.ng[g_axis][0]
        inner_slice = grid.get_inner_slice()

        inner_slice_1d = slice(ng_g_axis, -ng_g_axis if ng_g_axis > 0 else None)
        variable_1d_inner = variable_1d[inner_slice_1d]

        # Reshape for broadcasting based on the gravity axis
        reshape_dims = [1] * grid.ndim
        reshape_dims[g_axis] = -1  # e.g., (1, -1, 1) for g_axis=1
        variable_reshaped = variable_1d_inner.reshape(tuple(reshape_dims))

        # Broadcast to the full 3D inner domain shape
        if not nodal:
            target_shape_3d = variables.cell_vars[inner_slice + (VI.RHO,)].shape
        else:
            target_shape_3d = mpv.p2_nodes[inner_slice].shape
        variable_3D = np.broadcast_to(variable_reshaped, target_shape_3d)
        return variable_3D

    def initialize_solution(self, variables: "Variables", mpv: "MPV"):
        """
        Initialize solution using Exner-pressure generating function method.
        Constructs a balanced QG vortex by defining pressure first, then deriving
        velocities and thermodynamics to satisfy Geostrophic and Hydrostatic balance.
        """
        print("Initializing 3D QG Vortex (Exner-Consistent Method)...")

        grid = self.config.spatial_grid.grid
        thermo = Thermodynamics()
        thermo.update(self.config.global_constants.gamma)

        Msq = self.config.model_regimes.Msq
        g_vec = self.config.physics.gravity_strength
        g = g_vec[1]  # Y-gravity
        g_axis = 1
        f = self.config.physics.coriolis_strength[1]  # Y-coriolis
        Gamma = thermo.Gamma
        mpv.state(g_vec, Msq)

        # ------------------------------------------------------------------
        # 1. Setup Hydrostatic Background State on 1D Grid
        # ------------------------------------------------------------------
        strat_func = self.config.physics.stratification

        # A. Get 1D coordinates (full including ghosts) from the main grid
        y_cells_1d = grid.get_cell_coordinates(g_axis)
        y_nodes_1d = grid.get_node_coordinates(g_axis)

        # B. Evaluate stratification profile
        Y_cells_1d_val = strat_func(y_cells_1d)
        Y_nodes_1d_val = strat_func(y_nodes_1d)

        # C. Integrate for Hydrostatic Exner Pressure (pi0)
        # Reference index (first inner cell)
        ngy = grid.ng[1][0]
        dy = grid.dxyz[g_axis]

        # Node vars integration
        pi0_nodes_1d = np.zeros_like(y_nodes_1d)
        pi0_ref = 1.0 ** (thermo.gm1)  # p0=1.0
        pi0_nodes_1d[ngy] = pi0_ref

        # Integrate Up
        for j in range(ngy, len(y_nodes_1d) - 1):
            Y_val = Y_cells_1d_val[j]
            d_pi = -(Gamma * g / Y_val) * dy
            pi0_nodes_1d[j + 1] = pi0_nodes_1d[j] + d_pi
        # Integrate Down
        for j in range(ngy - 1, -1, -1):
            Y_val = Y_cells_1d_val[j]
            d_pi = (Gamma * g / Y_val) * dy
            pi0_nodes_1d[j] = pi0_nodes_1d[j + 1] + d_pi

        # Cell vars (Interpolate pi from nodes to cells for background)
        pi0_cells_1d = 0.5 * (pi0_nodes_1d[:-1] + pi0_nodes_1d[1:])

        # Derived thermodynamic vars
        p0_nodes_1d = pi0_nodes_1d ** thermo.Gammainv
        p0_cells_1d = pi0_cells_1d ** thermo.Gammainv
        rhoY0_cells_1d = pi0_cells_1d ** thermo.gm1inv
        rhoY0_nodes_1d = pi0_nodes_1d ** thermo.gm1inv

        # D. Assign to MPV Hydrostate (1D Variables)
        def assign_1d(target, source):
            target[:] = source

        # Cells
        assign_1d(mpv.hydrostate.cell_vars[..., HI.P2_0], pi0_cells_1d / Msq)
        assign_1d(mpv.hydrostate.cell_vars[..., HI.P0], p0_cells_1d)
        assign_1d(mpv.hydrostate.cell_vars[..., HI.Y0], Y_cells_1d_val)
        assign_1d(mpv.hydrostate.cell_vars[..., HI.S0], 1.0 / Y_cells_1d_val)
        assign_1d(mpv.hydrostate.cell_vars[..., HI.RHOY0], rhoY0_cells_1d)
        assign_1d(mpv.hydrostate.cell_vars[..., HI.RHO0], rhoY0_cells_1d / Y_cells_1d_val)

        # Nodes
        assign_1d(mpv.hydrostate.node_vars[..., HI.P2_0], pi0_nodes_1d / Msq)
        assign_1d(mpv.hydrostate.node_vars[..., HI.P0], p0_nodes_1d)
        assign_1d(mpv.hydrostate.node_vars[..., HI.Y0], Y_nodes_1d_val)
        assign_1d(mpv.hydrostate.node_vars[..., HI.S0], 1.0 / Y_nodes_1d_val)
        assign_1d(mpv.hydrostate.node_vars[..., HI.RHOY0], rhoY0_nodes_1d)
        assign_1d(mpv.hydrostate.node_vars[..., HI.RHO0], rhoY0_nodes_1d / Y_nodes_1d_val)

        # ------------------------------------------------------------------
        # 2. Initialize Variables on Cells (Vortex Perturbation)
        # ------------------------------------------------------------------
        inner_slice = grid.get_inner_slice()
        XC, YC, ZC = self._get_coordinates_3d(grid, inner_slice, nodal=False)

        # A. Get Background on Cells (Broadcasted from 1D arrays)
        Y0_inner = Y_cells_1d_val[ngy:-ngy]
        pi0_inner = pi0_cells_1d[ngy:-ngy]

        # Broadcast to 3D (1, ny, 1) -> (nx, ny, nz)
        Y0_c = Y0_inner.reshape(1, -1, 1)
        pi0_c = pi0_inner.reshape(1, -1, 1)

        # Calculate Background Exner gradient locally
        d_pi0_dy_c = - Gamma * g / Y0_c

        # B. Calculate Perturbation (Gaussian in Exner Pressure)
        H_scale = self.H_frac * (grid.y_end)
        pi_pert_c, d_pi_pert_dx_c, d_pi_pert_dy_c, d_pi_pert_dz_c = self._gaussian_perturbation(XC, YC, ZC, H_scale)

        # C. Construct Total Exner Pressure and Gradient
        # Note: pi_pert_c is the physical perturbation amplitude
        pi_total_c = pi0_c + pi_pert_c
        d_pi_total_dy_c = d_pi0_dy_c + d_pi_pert_dy_c

        # D. Recover Balanced Thermodynamics
        # Hydrostatic Balance: d(pi_tot)/dy = - Gamma * g / Y_tot
        # => Y_tot = - Gamma * g / (d_pi_tot/dy)
        Y_total_c = - Gamma * g / d_pi_total_dy_c

        # EOS: rhoY (Potential Pressure) = pi_tot ^ (1 / (gamma-1))
        rhoY_total_c = pi_total_c ** thermo.gm1inv

        # Density
        rho_total_c = rhoY_total_c / Y_total_c

        # E. Recover Balanced Geostrophic Velocities
        # Geostrophic Balance (Y vertical, X/Z horizontal):
        # f * k_hat x u = - Theta * Cp * Grad(Pi)
        # Components:
        # x-mom: -f * w = - (Y/Gamma) * dPi/dx  => w = (Y/(f*Gamma)) * dPi/dx
        # z-mom: +f * u = - (Y/Gamma) * dPi/dz  => u = -(Y/(f*Gamma)) * dPi/dz

        prefactor = Y0_c / (f * Gamma * Msq)
        # prefactor = Y_total_c / (f * Gamma * Msq)


        # Note signs: u relates to -dPi/dz, w relates to +dPi/dx for Cyclonic flow around Low
        if self.rossby_adjustment:
            u_geo_c = + prefactor * d_pi_pert_dz_c * self.rossby_adjustment
            w_geo_c = - prefactor * d_pi_pert_dx_c * self.rossby_adjustment
        else:
            u_geo_c = + prefactor * d_pi_pert_dz_c
            w_geo_c = - prefactor * d_pi_pert_dx_c

        u_total_c = self.u0 + u_geo_c
        w_total_c = self.w0 + w_geo_c
        v_total_c = np.zeros_like(u_total_c)  # Vertical velocity 0 initially

        # F. Assign to Cell Variables
        variables.cell_vars[inner_slice + (VI.RHO,)] = rho_total_c
        variables.cell_vars[inner_slice + (VI.RHOU,)] = rho_total_c * u_total_c
        variables.cell_vars[inner_slice + (VI.RHOV,)] = rho_total_c * v_total_c
        variables.cell_vars[inner_slice + (VI.RHOW,)] = rho_total_c * w_total_c
        variables.cell_vars[inner_slice + (VI.RHOY,)] = rhoY_total_c
        variables.cell_vars[inner_slice + (VI.RHOX,)] = 0.0

        # ------------------------------------------------------------------
        # 3. Initialize MPV on Nodes
        # ------------------------------------------------------------------
        XN, YN, ZN = self._get_coordinates_3d(grid, inner_slice, nodal=True)

        # Calculate perturbation on nodes
        pi_pert_n, _, _, _ = self._gaussian_perturbation(XN, YN, ZN, H_scale)

        rhoY0_nodes = self._reshape_hydrostate_into_3D(variables, mpv, rhoY0_nodes_1d, nodal=True)

        # Assign p2 (Exner perturbation / Msq)
        # Standard formulation: p2 represents the Exner pressure perturbation scaled by Msq.
        mpv.p2_nodes[inner_slice] = pi_pert_n / Msq
        # mpv.p2_nodes[inner_slice] = 0.0


        print("Initialization complete.")
        self.compute_diagnostics(variables, mpv)

    def compute_diagnostics(self, variables, mpv):
        """Check balance."""
        grid = self.config.spatial_grid.grid
        inner = grid.get_inner_slice()

        rho = variables.cell_vars[inner + (VI.RHO,)]
        u = variables.cell_vars[inner + (VI.RHOU,)] / rho

        max_u = np.max(np.abs(u))

        # Rossby Number Estimate
        Ro = max_u / (self.coriolis_strength * self.R0)
        print(f"DIAGNOSTICS: Max U={max_u:.4f}, Ro ~ {Ro:.4f} (Should be << 1 for QG)")
        print(f"DIAGNOSTICS: P_amp={self.p_amp}, Msq={self.config.model_regimes.Msq}")