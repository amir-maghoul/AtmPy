# import numpy as np
# from typing import TYPE_CHECKING
# from atmpy.test_cases.base_test_case import BaseTestCase
# from atmpy.configuration.simulation_configuration import SimulationConfig
# from atmpy.infrastructure.enums import (
#     BoundaryConditions as BdryType,
#     BoundarySide,
#     AdvectionRoutines,
#     RiemannSolvers,
#     FluxReconstructions,
#     LinearSolvers,
#     Preconditioners,
# )
# from atmpy.infrastructure.enums import (
#     VariableIndices as VI,
#     HydrostateIndices as HI,
#     SlopeLimiters as LimiterType,
# )
# from atmpy.physics.thermodynamics import Thermodynamics
# from atmpy.pressure_solver.discrete_operations import ClassicalDiscreteOperator
#
# if TYPE_CHECKING:
#     from atmpy.variables.variables import Variables
#     from atmpy.variables.multiple_pressure_variables import MPV
#
#
# class BaroclinicQGVortex(BaseTestCase):
#
#     def __init__(self, config_override: SimulationConfig = None):
#         _effective_config: SimulationConfig
#         run_setup_method = False
#
#         if config_override is not None:
#             _effective_config = config_override
#         else:
#             _effective_config = SimulationConfig()
#             run_setup_method = True
#
#         super().__init__(name="BaroclinicQGVortex", config=_effective_config)
#
#         ############################### Vortex Specific Parameters #####################################################
#         self.u0: float = 0.0  # Background velocity U (x)
#         self.v0: float = 0.0  # Background velocity V (y - vertical)
#         self.w0: float = 0.0  # Background velocity W (z)
#         self.p0: float = 1.0  # Background pressure
#
#         # Vortex Geometry
#         self.xc: float = 0.0
#         self.yc: float = 0.0
#         self.zc: float = 0.0
#
#         # Scales
#         self.size = 1.0
#         self.boxsize = 1.0 * self.size
#         self.h_scale = 1
#         self.y_boxscale = 0.1  # Height of domain relative to width
#         self.R0: float = 0.2 * self.size  # Horizontal radius
#         self.H_frac: float = 0.1  # Vertical aspect ratio of the vortex bubble
#
#         # Amplitude (Exner pressure perturbation amplitude)
#         # Needs to be small for QG validity (Rossby << 1)
#         self.p_amp: float = -1.0e-4
#
#         # Simulation settings
#         self.tmax = 1.0 * self.size
#         self.output_freq = 1
#         self.dtfixed0 = None
#         self.dtfixed = 0.001  # Conservative time step
#
#         # Physics
#         self.baroclinic = True
#         self.stratified_atmosphere = True
#         self.h_ref = 10_000.0
#         self.t_ref = 100.0
#         N_factor = 100
#         self.Nsq_ref = 1.0e-4 * N_factor
#
#         # Regimes
#         self.g = 1.0
#         self.coriolis_strength = 1000
#
#         # Switches for testing
#         self.is_nongeostrophic = 0
#         self.is_nonhydrostatic = 1
#         self.is_compressible = 0
#
#         if run_setup_method:
#             self.setup()
#
#     def setup(self):
#         """Configure the SimulationConfig for the 3D Traveling Vortex case."""
#         print("Setting up 3D Traveling Vortex QG configuration...")
#
#         #################################### Grid Configuration ########################################################
#         nx = nz = 40
#         ny = 10
#
#         # Domain extent
#         Lx = self.boxsize * self.h_scale
#         Ly = self.boxsize * self.y_boxscale
#
#         grid_updates = {
#             "ndim": 3,
#             "nx": nx,
#             "ny": ny,
#             "nz": nz,
#             "xmin": -Lx,
#             "xmax": Lx,
#             "ymin": -Ly,
#             "ymax": Ly,
#             "zmin": -Lx,
#             "zmax": Lx,
#             "ngx": 2,
#             "ngy": 2,
#             "ngz": 2,
#         }
#         self.set_grid_configuration(grid_updates)
#
#         #################################### Global Constants ##########################################################
#         constants_updates = {
#             "gamma": 1.4,
#             "R_gas": 287.4,
#             "p_ref": 1.0e5,
#             "T_ref": 300.0,
#             "h_ref": self.h_ref,
#             "t_ref": self.t_ref,
#             "Nsq_ref": self.Nsq_ref,
#             "grav": 10.0,
#         }
#         self.set_global_constants(constants_updates)
#
#         #################################### Boundary Conditions #######################################################
#         self.set_boundary_condition(BoundarySide.LEFT, BdryType.PERIODIC, mpv_type=BdryType.PERIODIC)
#         self.set_boundary_condition(BoundarySide.RIGHT, BdryType.PERIODIC, mpv_type=BdryType.PERIODIC)
#         self.set_boundary_condition(BoundarySide.BOTTOM, BdryType.REFLECTIVE_GRAVITY, mpv_type=BdryType.WALL)
#         self.set_boundary_condition(BoundarySide.TOP, BdryType.REFLECTIVE_GRAVITY, mpv_type=BdryType.WALL)
#         self.set_boundary_condition(BoundarySide.FRONT, BdryType.PERIODIC, mpv_type=BdryType.PERIODIC)
#         self.set_boundary_condition(BoundarySide.BACK, BdryType.PERIODIC, mpv_type=BdryType.PERIODIC)
#
#         #################################### Temporal Setting ##########################################################
#         temporal_updates = {
#             "CFL": 0.9,
#             "dtfixed": self.dtfixed,
#             "dtfixed0": self.dtfixed0,
#             "tout": np.array([0.0, self.tmax]),
#             "tmax": self.tmax,
#             "stepmax": 100_000,
#             "use_acoustic_cfl": True,
#         }
#         self.set_temporal(temporal_updates)
#
#         #################################### Model Regimes #############################################################
#         regime_updates = {
#             "is_nongeostrophic": self.is_nongeostrophic,
#             "is_nonhydrostatic": self.is_nonhydrostatic,
#             "is_compressible": self.is_compressible,
#         }
#         self.set_model_regimes(regime_updates)
#
#         #################################### Numerics ##################################################################
#         numerics_updates = {
#             "limiter": LimiterType.AVERAGE,
#             "riemann_solver": RiemannSolvers.MODIFIED_HLL,
#             "reconstruction": FluxReconstructions.MODIFIED_MUSCL,
#             "first_order_advection_routine": AdvectionRoutines.FIRST_ORDER_RK,
#             "second_order_advection_routine": AdvectionRoutines.STRANG_SPLIT,
#             "linear_solver": LinearSolvers.BICGSTAB,
#             "preconditioner": Preconditioners.DIAGONAL,
#             "initial_projection": True,
#         }
#         self.set_numerics(numerics_updates)
#
#         ################################### Physics Settings ###########################################################
#         g = self.g
#         cor = [0.0, self.coriolis_strength, 0.0]
#
#         physics_updates = {
#             "wind_speed": [self.u0, self.v0, self.w0],
#             "gravity_strength": (0.0, g, 0.0),
#             "coriolis_strength": cor,
#             "stratification": self.qg_stratification,
#         }
#         self.set_physics(physics_updates)
#
#         #################################### Outputs ###################################################################
#         output_updates = {
#             "output_type": "test",
#             "output_folder": "baroclinic_qg_vortex",
#             "output_base_name": "_baroclinic_qg_vortex",
#             "output_timesteps": True,
#             "output_frequency_steps": self.output_freq,
#         }
#         self.set_outputs(output_updates)
#
#         #################################### Diagnostics ###############################################################
#         diag_updates = {
#             "diag": True,
#             "diag_current_run": "atmpy_baroclinic_qg_vortex",
#         }
#         self.set_diagnostics(diag_updates)
#
#         self._update_Msq()
#         self._update_output_suffix()
#         print(f"Config done. Msq={self.config.model_regimes.Msq}, f={self.coriolis_strength}")
#
#     def qg_stratification(self, y: np.ndarray):
#         """Analytical stratification profile (N^2 = constant)."""
#         gl = self.config.global_constants
#         g_eff = self.config.physics.gravity_strength[1] / self.config.model_regimes.Msq
#         exponent = (self.Nsq_ref * gl.t_ref ** 2) * y / g_eff
#         return np.exp(exponent)
#
#     def _get_coordinates_3d(self, grid, inner_slice, nodal=False):
#         """Helper to get 3D coordinate meshgrids."""
#         if nodal:
#             # Check if we need full grid or sliced grid.
#             # For initialization we often want the full grid to handle BCs naturally.
#             if inner_slice == slice(None):
#                 x = grid.x_nodes
#                 y = grid.y_nodes
#                 z = grid.z_nodes
#             else:
#                 x = grid.x_nodes[inner_slice[0]]
#                 y = grid.y_nodes[inner_slice[1]]
#                 z = grid.z_nodes[inner_slice[2]]
#         else:
#             if inner_slice == slice(None):
#                 x = grid.x_cells
#                 y = grid.y_cells
#                 z = grid.z_cells
#             else:
#                 x = grid.x_cells[inner_slice[0]]
#                 y = grid.y_cells[inner_slice[1]]
#                 z = grid.z_cells[inner_slice[2]]
#
#         return np.meshgrid(x, y, z, indexing="ij")
#
#     def _gaussian_perturbation(self, X, Y, Z, H_scale):
#         """Calculates Gaussian perturbation (Scalar field only)."""
#         Lx = self.config.spatial_grid.grid.x_end - self.config.spatial_grid.grid.x_start
#         Lz = self.config.spatial_grid.grid.z_end - self.config.spatial_grid.grid.z_start
#
#         # Periodicity handling
#         dx = X - self.xc
#         dy = Y - self.yc
#         dz = Z - self.zc
#         dx = dx - Lx * np.round(dx / Lx)
#         dz = dz - Lz * np.round(dz / Lz)
#
#         r2_h = dx ** 2 + dz ** 2
#         envelope = np.exp(- (r2_h / (2 * self.R0 ** 2)) - (dy ** 2 / (2 * H_scale ** 2)))
#
#         val = self.p_amp * envelope
#         return val
#
#     def initialize_solution(self, variables: "Variables", mpv: "MPV"):
#         """
#         Initialize solution using Discrete Hydrostatic Balance.
#
#         Methodology:
#         1. Define Pressure Perturbation on NODES.
#         2. Define Pressure Background on NODES.
#         3. Add them to get Total Pressure on NODES.
#         4. Use ClassicalDiscreteOperator (the solver's own gradient tool) to compute
#            gradient(Total_Pressure_Nodes) -> Gradient_on_Cells.
#         5. Set Y (inverse theta) on Cells such that: Y * dP/dy + g = 0 EXACTLY.
#         """
#         print("Initializing 3D QG Vortex (Discrete Hydrostatic Balance)...")
#
#         # 1. Setup Tools
#         grid = self.config.spatial_grid.grid
#         thermo = Thermodynamics()
#         thermo.update(self.config.global_constants.gamma)
#
#         # Instantiate the discrete operator for gradient calculations
#         discrete_op = ClassicalDiscreteOperator(grid, boundary_manager=None)
#
#         Msq = self.config.model_regimes.Msq
#         g_vec = self.config.physics.gravity_strength
#         g = g_vec[1]  # Y-gravity is dominant
#         f = self.config.physics.coriolis_strength[1]
#         Gamma = thermo.Gamma
#         mpv.state(g_vec, Msq)
#
#         # ------------------------------------------------------------------
#         # A. Prepare 1D Background State (Analytical integration)
#         # ------------------------------------------------------------------
#         # We calculate this on 1D nodes first
#         strat_func = self.config.physics.stratification
#         g_axis = 1
#
#         y_cells_1d = grid.get_cell_coordinates(g_axis)
#         y_nodes_1d = grid.get_node_coordinates(g_axis)
#         ngy = grid.ng[1][0]
#         dy = grid.dxyz[g_axis]
#
#         Y_cells_1d_val = strat_func(y_cells_1d)
#         Y_nodes_1d_val = strat_func(y_nodes_1d)
#
#         # Integrate for Hydrostatic Exner Pressure (pi0)
#         pi0_nodes_1d = np.zeros_like(y_nodes_1d)
#         pi0_ref = 1.0 ** (thermo.gm1)
#         pi0_nodes_1d[ngy] = pi0_ref
#
#         for j in range(ngy, len(y_nodes_1d) - 1):
#             pi0_nodes_1d[j + 1] = pi0_nodes_1d[j] - (Gamma * g / Y_cells_1d_val[j]) * dy
#         for j in range(ngy - 1, -1, -1):
#             pi0_nodes_1d[j] = pi0_nodes_1d[j + 1] + (Gamma * g / Y_cells_1d_val[j]) * dy
#
#         # Fill MPV background variables (Used for reference, not for balance enforcement)
#         pi0_cells_1d = 0.5 * (pi0_nodes_1d[:-1] + pi0_nodes_1d[1:])
#         p0_cells_1d = pi0_cells_1d ** thermo.Gammainv
#         rhoY0_cells_1d = pi0_cells_1d ** thermo.gm1inv
#
#         p0_nodes_1d = pi0_nodes_1d ** thermo.Gammainv
#         rhoY0_nodes_1d = pi0_nodes_1d ** thermo.gm1inv
#
#         # Assign 1D arrays to MPV
#         def assign_1d(target, source):
#             target[:] = source
#
#         assign_1d(mpv.hydrostate.cell_vars[..., HI.P2_0], pi0_cells_1d / Msq)
#         assign_1d(mpv.hydrostate.cell_vars[..., HI.P0], p0_cells_1d)
#         assign_1d(mpv.hydrostate.cell_vars[..., HI.Y0], Y_cells_1d_val)
#         assign_1d(mpv.hydrostate.cell_vars[..., HI.S0], 1.0 / Y_cells_1d_val)
#         assign_1d(mpv.hydrostate.cell_vars[..., HI.RHOY0], rhoY0_cells_1d)
#         assign_1d(mpv.hydrostate.cell_vars[..., HI.RHO0], rhoY0_cells_1d / Y_cells_1d_val)
#
#         assign_1d(mpv.hydrostate.node_vars[..., HI.P2_0], pi0_nodes_1d / Msq)
#         assign_1d(mpv.hydrostate.node_vars[..., HI.P0], p0_nodes_1d)
#         assign_1d(mpv.hydrostate.node_vars[..., HI.Y0], Y_nodes_1d_val)
#         assign_1d(mpv.hydrostate.node_vars[..., HI.S0], 1.0 / Y_nodes_1d_val)
#         assign_1d(mpv.hydrostate.node_vars[..., HI.RHOY0], rhoY0_nodes_1d)
#         assign_1d(mpv.hydrostate.node_vars[..., HI.RHO0], rhoY0_nodes_1d / Y_nodes_1d_val)
#
#         # ------------------------------------------------------------------
#         # B. Construct Full 3D Nodal Fields (Nodes -> Cells)
#         # ------------------------------------------------------------------
#         # We need the full grid (including ghosts) to compute gradients for all cells
#         XN_full, YN_full, ZN_full = self._get_coordinates_3d(grid, slice(None), nodal=True)
#
#         # 1. Broadcast Background Pi0 to 3D Nodes
#         # pi0_nodes_1d has shape (Ny+1). We need (Nx+1, Ny+1, Nz+1)
#         # Assuming axis 1 is Y.
#         pi0_nodes_3d = pi0_nodes_1d.reshape(1, -1, 1)
#         pi0_nodes_3d = np.broadcast_to(pi0_nodes_3d, XN_full.shape).astype(np.float64)
#
#         # 2. Calculate Perturbation on Nodes
#         H_scale = self.H_frac * (grid.y_end - grid.y_start)
#         pi_pert_nodes_3d = self._gaussian_perturbation(XN_full, YN_full, ZN_full, H_scale)
#
#         # 3. Total Exner Pressure on Nodes
#         pi_total_nodes_3d = pi0_nodes_3d + pi_pert_nodes_3d
#
#         # ------------------------------------------------------------------
#         # C. Calculate Gradients using Discrete Operator
#         # ------------------------------------------------------------------
#         # discreteOp.gradient takes shape (Nx+1, Ny+1, Nz+1) -> returns (Nx, Ny, Nz) (Cells)
#         # This matches the behavior of variables.cell_vars exactly.
#
#         grad_pi_x, grad_pi_y, grad_pi_z = discrete_op.gradient(pi_total_nodes_3d)
#
#         # We also need gradients of the perturbation for velocity
#         grad_pert_x, grad_pert_y, grad_pert_z = discrete_op.gradient(pi_pert_nodes_3d)
#
#         # ------------------------------------------------------------------
#         # D. Derive Thermodynamic State (Y) from Vertical Gradient
#         # ------------------------------------------------------------------
#         # Enforce Discrete Hydrostatic Balance:
#         # Y_total * d(pi_total)/dy = - Gamma * g
#         # => Y_total = - Gamma * g / d(pi_total)/dy
#
#         # Note: grad_pi_y is defined on Cell Centers (size matches grid.cshape)
#         Y_total_cells = - (Gamma * g) / grad_pi_y
#
#         # Calculate other thermodynamic variables based on this balanced Y
#         # We need Pi_total on cells. Average from nodes (approximate for EOS, but exact for balance)
#         # Note: We could use the discrete operator's averaging logic if it existed,
#         # but simple averaging is standard for EOS.
#         pi_total_cells = 0.125 * (
#                 pi_total_nodes_3d[:-1, :-1, :-1] + pi_total_nodes_3d[1:, :-1, :-1] +
#                 pi_total_nodes_3d[:-1, 1:, :-1] + pi_total_nodes_3d[1:, 1:, :-1] +
#                 pi_total_nodes_3d[:-1, :-1, 1:] + pi_total_nodes_3d[1:, :-1, 1:] +
#                 pi_total_nodes_3d[:-1, 1:, 1:] + pi_total_nodes_3d[1:, 1:, 1:]
#         )
#
#         rhoY_total_cells = pi_total_cells ** thermo.gm1inv
#         rho_total_cells = rhoY_total_cells / Y_total_cells
#
#         # ------------------------------------------------------------------
#         # E. Derive Velocity (Geostrophic / Divergence Free)
#         # ------------------------------------------------------------------
#         # Use Background Y0 for the streamfunction relationship to ensure div(u)=0 analytically.
#         # u ~ d(psi)/dz, w ~ -d(psi)/dx
#         # psi ~ pi_pert * Y0 / (f * Gamma)
#
#         # Get Y0 on cells (broadcasted)
#         Y0_cells_1d = mpv.hydrostate.cell_vars[..., HI.Y0]
#         Y0_cells_3d = np.broadcast_to(Y0_cells_1d.reshape(1, -1, 1), grad_pert_z.shape)
#
#         prefactor = Y0_cells_3d / (f * Gamma)
#
#         # Use the discrete gradients of perturbation calculated earlier
#         u_geo = + prefactor * grad_pert_z / Msq
#         w_geo = - prefactor * grad_pert_x / Msq
#
#         u_total = self.u0 + u_geo
#         w_total = self.w0 + w_geo
#         v_total = np.zeros_like(u_total)
#
#         # ------------------------------------------------------------------
#         # F. Assign to Variables
#         # ------------------------------------------------------------------
#         # Note: grad_pi returns full cell shape (including ghosts).
#         # variables.cell_vars also has this shape. We can assign directly.
#
#         cell_vars = variables.cell_vars
#
#         cell_vars[..., VI.RHO] = rho_total_cells
#         cell_vars[..., VI.RHOU] = rho_total_cells * u_total
#         cell_vars[..., VI.RHOV] = rho_total_cells * v_total
#         cell_vars[..., VI.RHOW] = rho_total_cells * w_total
#         cell_vars[..., VI.RHOY] = rhoY_total_cells
#         cell_vars[..., VI.RHOX] = 0.0
#
#         # ------------------------------------------------------------------
#         # G. Assign p2_nodes
#         # ------------------------------------------------------------------
#         # mpv.p2_nodes must match the nodal perturbation used to generate the gradients.
#         # pi_pert_nodes_3d covers the full grid.
#         # Check shape of mpv.p2_nodes vs pi_pert_nodes_3d
#         mpv.p2_nodes[...] = pi_pert_nodes_3d / Msq
#
#         print("Initialization complete. Vertical force balance matches discrete operator exactly.")
#         self.compute_diagnostics(variables, mpv)
#
#     def compute_diagnostics(self, variables, mpv):
#         """Check balance."""
#         grid = self.config.spatial_grid.grid
#         inner = grid.get_inner_slice()
#
#         rho = variables.cell_vars[inner + (VI.RHO,)]
#         u = variables.cell_vars[inner + (VI.RHOU,)] / rho
#
#         max_u = np.max(np.abs(u))
#         Ro = max_u / (self.coriolis_strength * self.R0)
#         print(f"DIAGNOSTICS: Max U={max_u:.4f}, Ro ~ {Ro:.4f}")

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
        self.y_boxscale = 0.1  # Height of domain relative to width
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

        self.g = 1.0
        self.coriolis_strength =  10 # f parameter (Make relatively large for small Ro)

        # Switches for testing (User wants code to work with these = 1)
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
            "output_folder": "baroclinic_qg_vortex",
            "output_base_name": "_baroclinic_qg_vortex",
            "output_timesteps": True,
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
        """
        Initialize solution using Exner-pressure generating function method.

        CRITICAL FIX:
        We explicitly calculate the hydrostatic background state on the 1D vertical grid
        and assign it to `mpv.hydrostate`. This ensures `S0` has the correct vertical gradient,
        so `dChi` is non-zero. This prevents the Coriolis matrix determinant from vanishing
        when `is_nonhydrostatic=False`.
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
        # We use get_coordinates because grid.y_cells might not exist on all grid types
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
        # mpv.hydrostate variables are on a 1D grid, so they are 1D arrays (or (N,1) depending on impl)
        # The fix: Assign assuming 1D matching source size

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
        # We slice to inner domain for broadcasting
        Y0_inner = Y_cells_1d_val[ngy:-ngy]
        pi0_inner = pi0_cells_1d[ngy:-ngy]

        # Broadcast to 3D (1, ny, 1) -> (nx, ny, nz)
        # Note: shape depends on axis. For y-axis (1), reshape to (1, -1, 1)
        Y0_c = Y0_inner.reshape(1, -1, 1)
        pi0_c = pi0_inner.reshape(1, -1, 1)

        # Calculate Background Exner gradient locally for balance check
        d_pi0_dy_c = - Gamma * g / Y0_c

        # B. Calculate Perturbation (Gaussian in Exner Pressure)
        H_scale = self.H_frac * (grid.y_end - grid.y_start)
        pi_pert_c, d_pi_pert_dx_c, d_pi_pert_dy_c, d_pi_pert_dz_c = self._gaussian_perturbation(XC, YC, ZC, H_scale)

        # C. Construct Total Exner Pressure and Gradient
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
        # u_geo = + (Y / (f * Gamma)) * d(pi_tot)/dz
        # w_geo = - (Y / (f * Gamma)) * d(pi_tot)/dx

        prefactor = Y_total_c / (f * Gamma)
        # prefactor = Y0_c / (f * Gamma)
        u_geo_c = + prefactor * d_pi_pert_dz_c / Msq
        w_geo_c = - prefactor * d_pi_pert_dx_c / Msq


        eps = 0.0 # geostrophic imbalance factor
        u_total_c = self.u0 + (1 + eps) * u_geo_c
        w_total_c = self.w0 + (1 + eps) * w_geo_c
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

        eps = 0.5
        # Assign p2 (Exner perturbation / Msq)
        mpv.p2_nodes[inner_slice] = (1 + eps) * pi_pert_n / Msq

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