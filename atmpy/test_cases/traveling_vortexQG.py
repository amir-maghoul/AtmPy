import numpy as np
from dataclasses import field
from typing import List, Tuple, Dict, Any, TYPE_CHECKING
import logging
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


class TravelingVortexQG(BaseTestCase):

    def __init__(self, config_override: SimulationConfig = None):
        _effective_config: SimulationConfig
        run_setup_method = False

        if config_override is not None:
            _effective_config = config_override
        else:
            _effective_config = SimulationConfig()
            run_setup_method = True

        super().__init__(name="TravelingVortexQG", config=_effective_config)

        ############################### Vortex Specific Parameters #####################################################
        self.u0: float = 0.0  # Background velocity U (x)
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
        self.h_scale = 1
        self.y_boxscale = 0.2  # Height of domain relative to width
        self.R0: float = 0.2 * self.size  # Horizontal radius
        self.H_frac: float = 0.1 # Vertical aspect ratio of the vortex bubble

        # Amplitude (Exner pressure perturbation amplitude)
        # Needs to be small for QG validity (Rossby << 1)
        self.p_amp: float = -1.0e-4

        # Simulation settings
        self.tmax = 1.0 * self.size
        self.output_freq = 1
        self.dtfixed0 = None
        self.dtfixed = 0.001  # Conservative time step

        # Physics
        self.baroclinic = True
        self.stratified_atmosphere = True
        self.h_ref = 10_000.0
        self.t_ref = 100.0
        N_factor = 10
        self.Nsq_ref = 1.0e-4*N_factor

        # Regimes
        # Msq is calculated automatically by the solver config,
        # but we set the scaling reference here.
        # self.Msq = 1.0e-1  # Target Mach squared magnitude

        self.g = 1.0
        self.coriolis_strength =  1000  # f parameter (Make relatively large for small Ro)

        # Switches for testing (User wants code to work with these = 1)
        self.is_nongeostrophic = 0
        self.is_nonhydrostatic = 1
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

        #################################### Temporal Setting ##########################################################
        temporal_updates = {
            "CFL": 0.9,
            "dtfixed": self.dtfixed,
            "dtfixed0": self.dtfixed0,
            "tout": np.array([0.0, self.tmax]),
            "tmax": self.tmax,
            "stepmax": 100_000,
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
            "initial_projection": True,  # Projects initial velocity to be divergence free if needed
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
            "output_folder": "traveling_vortex_qg",
            "output_base_name": "_traveling_vortex_qg",
            "output_timesteps": True,
            "output_frequency_steps": self.output_freq,
        }
        self.set_outputs(output_updates)

        #################################### Diagnostics ###############################################################
        diag_updates = {
            "diag": True,
            "diag_current_run": "atmpy_traveling_vortex_qg",
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
        """Calculates Gaussian perturbation and its spatial derivatives."""
        Lx = self.config.spatial_grid.grid.x_end - self.config.spatial_grid.grid.x_start
        Lz = self.config.spatial_grid.grid.z_end - self.config.spatial_grid.grid.z_start

        # Periodicity handling
        dx = X - self.xc
        dy = Y - self.yc
        dz = Z - self.zc
        dx = dx - Lx * np.round(dx / Lx)
        dz = dz - Lz * np.round(dz / Lz)

        r2_h = dx ** 2 + dz ** 2
        envelope = np.exp(- (r2_h / (2 * self.R0 ** 2)) - (dy ** 2 / (2 * H_scale ** 2)))

        val = self.p_amp * envelope

        # Derivatives
        d_val_dx = val * (-dx / self.R0 ** 2)
        d_val_dy = val * (-dy / H_scale ** 2)
        d_val_dz = val * (-dz / self.R0 ** 2)

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
        grid = self.config.spatial_grid.grid
        thermo = Thermodynamics()
        thermo.update(self.config.global_constants.gamma)
        Gamma = thermo.Gamma
        gm1inv = thermo.gm1inv

        # FULL grid coordinates (including ghosts)
        XC = grid.x_cells  # shape (nx_total,)
        YC = grid.y_cells  # shape (ny_total,)
        ZC = grid.z_cells  # shape (nz_total,)

        XN = grid.x_nodes
        YN = grid.y_nodes
        ZN = grid.z_nodes

        XC3, YC3, ZC3 = np.meshgrid(XC, YC, ZC, indexing='ij')  # full 3D cell centers
        XN3, YN3, ZN3 = np.meshgrid(XN, YN, ZN, indexing='ij')  # full 3D nodes

        # Background stratification (your exponential profile)
        Y_bg_cells = self.config.physics.stratification(YC3)
        Y_bg_nodes = self.config.physics.stratification(YN3)

        # Small localized perturbation in potential temperature
        amp = 0.001
        H_scale = self.H_frac * (YC[-1] - YC[0])
        Lx = XC[-1] - XC[0]
        Lz = ZC[-1] - ZC[0]
        LxN = XN[-1] - XN[0]
        LzN = ZN[-1] - ZN[0]

        dx_c = XC3 - self.xc;
        dx_c -= Lx * np.round(dx_c / Lx)
        dz_c = ZC3 - self.zc;
        dz_c -= Lz * np.round(dz_c / Lz)
        dx_n = XN3 - self.xc;
        dx_n -= LxN * np.round(dx_n / LxN)
        dz_n = ZN3 - self.zc;
        dz_n -= LzN * np.round(dz_n / LzN)

        pert_c = amp * np.exp(-(dx_c ** 2 + dz_c ** 2) / (self.R0 ** 2) - (YC3 - self.yc) ** 2 / (H_scale ** 2))
        pert_n = amp * np.exp(-(dx_n ** 2 + dz_n ** 2) / (self.R0 ** 2) - (YN3 - self.yc) ** 2 / (H_scale ** 2))

        Y_cells_full = Y_bg_cells + pert_c
        Y_nodes_full = Y_bg_nodes + pert_n

        # NOW it works
        HySt = mpv.compute_hydrostate_from_profile3D(
            Y_cells_full, Y_nodes_full,
            self.config.physics.gravity_strength,
            self.config.model_regimes.Msq
        )

        # Extract only inner part for prognostic variables
        inner = grid.get_inner_slice()
        rhoY0_inner = HySt.rhoY0 # shape (nx, ny, nz)
        rho_inner = rhoY0_inner / Y_cells_full

        # Geostrophic velocity (thermal wind)
        f = self.config.physics.coriolis_strength[1]
        g = self.config.physics.gravity_strength[1]

        dpert_dz = np.diff(pert_c, axis=2) / grid.dz
        dpert_dx = np.diff(pert_c, axis=0) / grid.dx
        # simple central diff + padding omitted for brevity

        u_g = np.zeros_like(Y_bg_cells)
        w_g = np.zeros_like(Y_bg_cells)
        u_g[:, :, :-1] = (g / f) * dpert_dz / Y_bg_cells[:, :, :-1]
        w_g[:-1, :, :] = -(g / f) * dpert_dx / Y_bg_cells[:-1, :, :]

        variables.cell_vars[..., VI.RHO] = rho_inner
        variables.cell_vars[..., VI.RHOY] = rhoY0_inner
        variables.cell_vars[..., VI.RHOU] = rho_inner * u_g
        variables.cell_vars[..., VI.RHOW] = rho_inner * w_g
        variables.cell_vars[..., VI.RHOV] = 0.0

        # p2 from hydrostatic integration
        mpv.p2_nodes[...] = HySt.p20_nodes  # already full size

        mpv.correct_initial_pressure3D(variables, self.config.model_regimes.Msq, g)