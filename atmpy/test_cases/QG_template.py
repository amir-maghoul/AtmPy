import numpy as np
from dataclasses import field
from typing import List, Tuple, Dict, Any, TYPE_CHECKING
import matplotlib
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

def gradient_3d(
    p: np.ndarray, dx: float, dy: float, dz: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ 3D gradient (cell-centered).

    Parameters
    ----------
    p : np.ndarray of shape (nnx, nny, nnz)
        The input scalar function (mostly pressure or exner pressure perturbation) defined on nodes.
    dx, dy, dz : float
        The discretization fineness in each coordinate direction

    Returns
    -------
    Dpx, Dpy, Dpz : np.ndarray of shape (nnx-1, nny-1, nnz-1)
        The derivative in each coordinate direction
    """
    nx = p.shape[0] - 1
    ny = p.shape[1] - 1
    nz = p.shape[2] - 1

    # Preallocate for derivatives
    Dpx = np.empty((nx, ny, nz), dtype=p.dtype)
    Dpy = np.empty((nx, ny, nz), dtype=p.dtype)
    Dpz = np.empty((nx, ny, nz), dtype=p.dtype)

    inv_dx_quarter = 0.25 / dx
    inv_dy_quarter = 0.25 / dy
    inv_dz_quarter = 0.25 / dz

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Nodal values surrounding the cell (i, j, k) center
                p000 = p[i, j, k]
                p100 = p[i + 1, j, k]
                p010 = p[i, j + 1, k]
                p001 = p[i, j, k + 1]
                p110 = p[i + 1, j + 1, k]
                p101 = p[i + 1, j, k + 1]
                p011 = p[i, j + 1, k + 1]
                p111 = p[i + 1, j + 1, k + 1]

                # Average gradient in x across the cell
                Dpx[i, j, k] = (
                    (p100 - p000) + (p110 - p010) + (p101 - p001) + (p111 - p011)
                ) * inv_dx_quarter
                # Average gradient in y across the cell
                Dpy[i, j, k] = (
                    (p010 - p000) + (p110 - p100) + (p011 - p001) + (p111 - p101)
                ) * inv_dy_quarter
                # Average gradient in z across the cell
                Dpz[i, j, k] = (
                    (p001 - p000) + (p101 - p100) + (p011 - p010) + (p111 - p110)
                ) * inv_dz_quarter
    return Dpx, Dpy, Dpz

class TravelingVortexQG(BaseTestCase):

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

        super().__init__(name="TravelingVortexQG", config=_effective_config)

        ############################### Vortex Specific Parameters #####################################################

        self.u0: float = 0.0  # Background velocity U
        self.v0: float = 0.0  # Background velocity V
        self.w0: float = 0.0  # Background velocity W
        self.p0: float = 1.0  # Background pressure (dimensionless)

        self.rho0: float = 0.5  # Background density (dimensionless)
        self.del_rho: float = 0.5  # Density deficit at center
        self.alpha = 0
        self.alpha_const = 1

        self.xc: float = 0.0  # Vortex center x
        self.yc: float = 0.0  # Vortex center y
        self.zc: float = 0.0  # Vortex center z

        self.rotdir: float = 1.0  # Rotation direction
        self.size = 1
        self.boxsize = 1*self.size
        self.y_boxscale = 0.1
        self.R0: float = 0.25*self.size  # Vortex radius scale
        self.tmax = 1.0*self.size

        self.p_amp: float = -1.0e-5 # nondimensional pressure perturbation amplitude
        self.H_frac: float = 0.3  # fraction of Ly used for vertical scale H
        self.del_Y: float = 0.005  # Potential temperature perturbation amplitude
        N_factor = 10000
        self.Nsq_ref = 1.0e-4*N_factor
        Msq_factor = 100
        self.Msq = 1.0/Msq_factor

        self.baroclinic = True
        self.stratified_atmosphere = True

        self.dtfixed0 = None
        self.dtfixed = self.dtfixed0 if self.dtfixed0 else 0.1
        self.acoustic_cfl = True

        self.output_freq = 1
        self.h_ref = 10_000.0
        self.t_ref = 100.0
        self.g = 1.0 if self.stratified_atmosphere else 0.0
        coriolis_factor = 100
        force = 1.0e-4 * self.t_ref * coriolis_factor  # Constant coriolis force
        self.cor_f = [0.0, 0.0, 0.0]
        self.coriolis_axis = 1
        self.cor_f[self.coriolis_axis] = force

        self.is_nongeostrophic = 0
        self.is_nonhydrostatic = 0
        self.is_compressible = 1


        if run_setup_method:
            self.setup()

    def setup(self):
        """Configure the SimulationConfig for the 3D Traveling Vortex case."""
        print("Setting up 3D Traveling Vortex configuration...")

        #################################### Grid Configuration ########################################################
        nx = nz = 40
        ny = 40

        boxsize = self.boxsize # This should be always less than 1.0 otherwise the stratification creates inf due to
                      # large dx.

        grid_updates = {
            "ndim": 3,
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "xmin": -boxsize,
            "xmax": boxsize,
            "ymin": -boxsize*self.y_boxscale,
            "ymax": boxsize*self.y_boxscale,
            "zmin": -boxsize,
            "zmax": boxsize,
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
            "grav": 10,
        }
        self.set_global_constants(constants_updates)

        #################################### Boundary Conditions #######################################################
        # Periodic in X and Z, Wall in Y (vertical)
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
            BoundarySide.FRONT, BdryType.PERIODIC, mpv_type=BdryType.PERIODIC
        )
        self.set_boundary_condition(
            BoundarySide.BACK, BdryType.PERIODIC, mpv_type=BdryType.PERIODIC
        )

        #################################### Temporal Setting ##########################################################
        temporal_updates = {
            "CFL": 0.9,
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
            "Msq": self.Msq,
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
            "initial_projection": True,
        }
        self.set_numerics(numerics_updates)

        ################################### Physics Settings ###########################################################
        cor = self.cor_f
        g = self.g
        physics_updates = {
            "wind_speed": [self.u0, self.v0, self.w0],
            "gravity_strength": (0.0, g, 0.0),  # Gravity is zero for this test case
            "coriolis_strength": cor,
            "stratification": self.qg_stratification if self.baroclinic else self.constant_stratification,
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
            "diag_current_run": "atmpy_travelling_vortex_3d",
        }
        self.set_diagnostics(diag_updates)

        ################################################################################################################
        if "Msq" not in regime_updates:
            self._update_Msq()
        self._update_output_suffix()

        print(f"Configuration complete. Msq = {self.config.model_regimes.Msq}")
        print(
            f"Output files: {self.config.outputs.output_base_name}{self.config.outputs.output_suffix}"
        )

    def qg_stratification(self, y: np.ndarray):
        """Analytical stratification profile, matching project 1."""
        gl = self.config.global_constants
        g_eff = self.config.physics.gravity_strength[1] / self.config.model_regimes.Msq
        Nsq_scaled = gl.Nsq_ref * (gl.t_ref**2)
        # return np.exp(Nsq_scaled * y / g_eff)
        return np.exp(gl.Nsq_ref * y / g_eff)

    def constant_stratification(self, y: np.ndarray):
        """Constant stratification profile, matching project 1."""
        return 1.0

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

    def _reshape_hydrostate_into_3D_full(self, variables: "Variables", mpv: "MPV", variable_1d: np.ndarray,
                                    nodal: bool = False):
        """ Reshape the given hydrostate 1D profile into 3D by broadcasting to FULL grid shape. """
        g_axis = self.config.physics.gravity.axis
        grid = self.config.spatial_grid.grid

        # Use full 1D variable (including ghosts), no inner slice
        reshape_dims = [1] * grid.ndim
        reshape_dims[g_axis] = -1
        variable_reshaped = variable_1d.reshape(tuple(reshape_dims))

        # Broadcast to full 3D shape
        if not nodal:
            target_shape_3d = grid.cshape  # Full cell shape
        else:
            target_shape_3d = grid.nshape  # Full node shape
        variable_3D = np.broadcast_to(variable_reshaped, target_shape_3d)
        return variable_3D

    def initialize_solution(self, variables: "Variables", mpv: "MPV"):
        """Initialize density, momentum, potential temperature, and pressure fields for 3D.

        """
        print("Initializing solution for 3D Traveling Vortex...")

        grid = self.config.spatial_grid.grid
        thermo = Thermodynamics()
        thermo.update(self.config.global_constants.gamma)
        Msq = self.config.model_regimes.Msq

        # --- Calculate Hydrostatic Base State ---
        gravity = self.config.physics.gravity_strength
        g_axis = self.config.physics.gravity.axis
        g = gravity[g_axis]
        mpv.state(gravity, Msq)
        coriolis = self.config.physics.coriolis_strength[
            self.coriolis_axis
        ]  # Coriolis force in z-direction which is *NOT* the vertical direction

        # Get slice for the full inner domain
        inner_slice = grid.get_inner_slice()
        inner_slice_1d = inner_slice[g_axis]

        # --- Get Cell-Centered Coordinates ---
        if grid.ndim == 3:
            XC, YC, ZC = np.meshgrid(
                grid.x_cells[inner_slice[0]],
                grid.y_cells[inner_slice[1]],
                grid.z_cells[inner_slice[2]],
                indexing="ij",
            )
        else:
            raise NotImplementedError(
                "Traveling vortex initialization requires a 3D grid"
            )

        if grid.ndim == 3:
            XN, YN, ZN = np.meshgrid(
                grid.x_nodes[inner_slice[0]],
                grid.y_nodes[inner_slice[1]],
                grid.z_nodes[inner_slice[2]],
                indexing="ij",
            )
        else:
            raise NotImplementedError("Nodal calculation requires 3D grid")

        # --- Calculate Distance from Vortex Center in the XZ-Plane (Handling Periodicity) ---
        Lx = grid.x_end - grid.x_start
        Ly = grid.y_end - grid.y_start
        Lz = grid.z_end - grid.z_start

        dx = XC - self.xc
        dy = YC - self.yc
        dz = ZC - self.zc

        # Account for periodicity
        dx = dx - Lx * np.round(dx / Lx)
        dz = dz - Lz * np.round(dz / Lz)

        strat_func = self.config.physics.stratification

        if self.stratified_atmosphere:
            # Override with custom stratification profile for constant N^2
            cells_1d = grid.get_cell_coordinates(g_axis)
            nodes_1d = grid.get_node_coordinates(g_axis)
            Y_cells_1d = strat_func(cells_1d)
            Y_nodes_1d = strat_func(nodes_1d)
            Y_cells_nd = self._reshape_hydrostate_into_3D_full(variables, mpv, Y_cells_1d, nodal=False)
            Y_nodes_nd = self._reshape_hydrostate_into_3D_full(variables, mpv, Y_nodes_1d, nodal=True)
            HySt = mpv.compute_hydrostate_from_profile3D(Y_cells_nd, Y_nodes_nd, gravity, Msq)

            # Assign to hydrostate (extract 1D from 3D inner)
            ng_g = grid.ng[g_axis][0]
            inner_slice_1d = slice(ng_g, -ng_g)
            mpv.hydrostate.cell_vars[inner_slice_1d, HI.P2_0] = HySt.p20[0, :, 0]
            mpv.hydrostate.cell_vars[inner_slice_1d, HI.P0] = HySt.p0[0, :, 0]
            mpv.hydrostate.cell_vars[inner_slice_1d, HI.RHOY0] = HySt.rhoY0[0, :, 0]
            mpv.hydrostate.cell_vars[inner_slice_1d, HI.S0] = HySt.S0[0, :, 0]
            mpv.hydrostate.cell_vars[inner_slice_1d, HI.Y0] = Y_cells_1d[inner_slice_1d]
            mpv.hydrostate.cell_vars[inner_slice_1d, HI.RHO0] = mpv.hydrostate.cell_vars[inner_slice_1d, HI.RHOY0] / mpv.hydrostate.cell_vars[
                inner_slice_1d, HI.Y0]
            mpv.hydrostate.node_vars[inner_slice_1d, HI.P2_0] = HySt.p20_nodes[0, :, 0]
            # Similarly for other node vars, assuming similar structure

        # The vortex is uniform in y, so the radius is calculated only on the xz-plane
        r_cell = np.sqrt(dx**2 + dz**2)
        r_over_R0_cell = np.divide(r_cell, self.R0, where=self.R0 != 0)

        H = self.H_frac * Ly
        exponent = - ((r_over_R0_cell) ** 2)/2 - ((dy / H) ** 2)/2
        # dp2_cells = self.p_amp * np.exp(exponent)

        Y_pert_cell = self.del_Y * np.exp(exponent)

        # --- Get background fields broadcasted to 3D ---
        Y0_cells = self._reshape_hydrostate_into_3D_full(variables, mpv, mpv.hydrostate.cell_vars[..., HI.Y0])
        rhoY0_cells = self._reshape_hydrostate_into_3D_full(variables, mpv, mpv.hydrostate.cell_vars[..., HI.RHOY0])
        rho0_cells = self._reshape_hydrostate_into_3D_full(variables, mpv, mpv.hydrostate.cell_vars[..., HI.RHO0])

        # --- Calculate Total Potential Temperature and Density ---
        Y_total_cell = Y0_cells[inner_slice] + Y_pert_cell
        rho_total = rhoY0_cells[inner_slice] / Y_total_cell
        rho_pert = rho_total - rho0_cells[inner_slice]

        # --- Calculate Hydrostatic Pressure Perturbation for Cells ---
        p_pert_cells = mpv.integrate_density_hydrostatically(rho_pert, gravity)

        # --- Calculate Velocity Perturbations from Geostrophic Balance ---
        grad_p_x, _, grad_p_z = np.gradient(p_pert_cells, grid.dxyz[0], grid.dxyz[1], grid.dxyz[2])
        # grad_p_x, grad_p_z = np.zeros_like(rho_total), np.zeros_like(rho_total)
        # grad_p_x_inn, _, grad_p_z_inn = gradient_3d(p_pert_cells, grid.dxyz[0], grid.dxyz[1], grid.dxyz[2])
        # inner_slice_one_element = one_element_inner_slice(ndim=3, full=False)
        # grad_p_x[:-1, :-1, :-1] = grad_p_x_inn
        # grad_p_z[:-1, :-1, :-1] = grad_p_z_inn

        cor = coriolis
        u_pert = self.rotdir * (1.0 / (cor * rho_total)) * grad_p_z
        w_pert = self.rotdir * (-1.0 / (cor * rho_total)) * grad_p_x

        u_total = self.u0 + u_pert
        w_total = self.w0 + w_pert
        v_total = np.full_like(u_total, self.v0)  # background v only

        rhoY0_cells_1d_full = mpv.hydrostate.cell_vars[..., HI.RHOY0]
        rhoY0_cells = self._reshape_hydrostate_into_3D(variables, mpv, rhoY0_cells_1d_full)

        rhoX = rho_total * (1.0 / Y_total_cell - 1.0 / Y0_cells[inner_slice])

        # --- Assign to Cell Variables (Inner Domain Only) ---
        variables.cell_vars[inner_slice + (VI.RHO,)] = rho_total
        variables.cell_vars[inner_slice + (VI.RHOU,)] = rho_total * u_total
        variables.cell_vars[inner_slice + (VI.RHOV,)] = rho_total * v_total
        variables.cell_vars[inner_slice + (VI.RHOW,)] = rho_total * w_total
        variables.cell_vars[inner_slice + (VI.RHOY,)] = rhoY0_cells
        variables.cell_vars[inner_slice + (VI.RHOX,)] = rhoX


########################################################################################################################
######### --- Calculate Nodal Pressure Perturbation p2 (Nodes, Inner Domain Only) --- ##################################
        dx_node = XN - self.xc
        dy_node = YN - self.yc
        dz_node = ZN - self.zc
        dx_node = dx_node - Lx * np.round(dx_node / Lx)
        dz_node = dz_node - Lz * np.round(dz_node / Lz)

        r_node = np.sqrt(dx_node**2 + dz_node**2)  # Radius is in xz-plane
        r_over_R0_node = np.divide(r_node, self.R0, where=self.R0 != 0)
        mask_node = r_node < self.R0

        Y0_nodes_1d = mpv.hydrostate.node_vars[..., HI.Y0]
        Y0_nodes = self._reshape_hydrostate_into_3D(variables, mpv, Y0_nodes_1d, nodal=True)
        rhoY0_nodes_1d = mpv.hydrostate.node_vars[..., HI.RHOY0]
        rhoY0_nodes = self._reshape_hydrostate_into_3D(variables, mpv, rhoY0_nodes_1d, nodal=True)
        rho0_nodes_1d = mpv.hydrostate.node_vars[..., HI.RHO0]
        rho0_nodes = self._reshape_hydrostate_into_3D(variables, mpv,rho0_nodes_1d, nodal=True)

        H = self.H_frac * Ly
        Y_pert_node = self.del_Y * np.exp(- ((r_node / self.R0) ** 2)/2) * np.exp(
            - ((dy_node / H) ** 2)/2)

        Y_total_node = Y0_nodes + Y_pert_node
        rho_total_node = rhoY0_nodes / Y_total_node
        rho_pert_node = rho_total_node - rho0_nodes

        p0_nodes_1d = mpv.hydrostate.node_vars[..., HI.P0]
        p0_nodes = self._reshape_hydrostate_into_3D(variables, mpv, p0_nodes_1d, nodal=True)
        p_pert_nodes = mpv.integrate_density_hydrostatically(rho_pert_node, gravity)
        p2_nodes_unscaled = p_pert_nodes
        p0 = self.p0

        # convert to QG pressure variable p2 (nodes)
        if self.config.model_regimes.is_compressible:
            p_total_nodes = p0_nodes + Msq * p2_nodes_unscaled
            p_total_nodes_safe = np.maximum(p_total_nodes, 1e-9)
            rhoY0_nodes = p_total_nodes_safe**thermo.gamminv

        mpv.p2_nodes[inner_slice] = thermo.Gamma * np.divide(
            p2_nodes_unscaled, rhoY0_nodes, where=rhoY0_nodes != 0
        )

        compute_diagnostics(self, variables, mpv)


        logging.info("3D solution initialization complete.")

def compute_diagnostics(self, variables: "Variables", mpv: "MPV"):
    """
    Compute and print diagnostics for balance and field summaries.
    Call this after initialize_solution() to verify QG balances.
    """
    print("Running diagnostics for Traveling Vortex QG...")

    grid = self.config.spatial_grid.grid
    inner_slice = grid.get_inner_slice()
    g_axis = self.config.physics.gravity.axis  # Assuming y=1
    g = self.config.physics.gravity_strength[g_axis]
    cor = self.config.physics.coriolis_strength[self.coriolis_axis]  # f
    Msq = self.config.model_regimes.Msq

    # Extract key fields (from inner domain)
    rho_total = variables.cell_vars[inner_slice + (VI.RHO,)]
    u_total = variables.cell_vars[inner_slice + (VI.RHOU,)] / rho_total
    v_total = variables.cell_vars[inner_slice + (VI.RHOV,)] / rho_total
    w_total = variables.cell_vars[inner_slice + (VI.RHOW,)] / rho_total
    rhoY_total = variables.cell_vars[inner_slice + (VI.RHOY,)]
    Y_total = rhoY_total / rho_total  # Total potential temperature

    # Background fields (reshaped to 3D inner)
    rho0_1d = mpv.hydrostate.cell_vars[..., HI.RHO0]
    Y0_1d = mpv.hydrostate.cell_vars[..., HI.Y0]
    rho0 = self._reshape_hydrostate_into_3D(variables, mpv, rho0_1d)
    Y0 = self._reshape_hydrostate_into_3D(variables, mpv, Y0_1d)

    rho_pert = rho_total - rho0
    Y_pert = Y_total - Y0

    # Pressure perturbation (from your init; assume you store it or recompute if needed)
    # For demo, recompute p_pert_cells as in init (or pass it as arg)
    p_pert_cells = mpv.integrate_density_hydrostatically(rho_pert, self.config.physics.gravity_strength)

    # Compute gradients (using np.gradient for simplicity; assumes uniform grid)
    grad_p_x, grad_p_y, grad_p_z = np.gradient(p_pert_cells, *grid.dxyz)
    grad_Y_x, grad_Y_y, grad_Y_z = np.gradient(Y_pert, *grid.dxyz)
    grad_u_x, grad_u_y, grad_u_z = np.gradient(u_total, *grid.dxyz)
    grad_v_x, grad_v_y, grad_v_z = np.gradient(v_total, *grid.dxyz)
    grad_w_x, grad_w_y, grad_w_z = np.gradient(w_total, *grid.dxyz)
    div = grad_u_x + grad_v_y + grad_w_z
    max_div = np.max(np.abs(div))
    mean_div = np.mean(np.abs(div))
    print(f"Divergence - Max: {max_div:.2e}, Mean: {mean_div:.2e}")
    print(f"grad_u_x - Max: {np.max(np.abs(grad_u_x)):.2e}, Mean: {np.mean(np.abs(grad_u_x)):.2e}")
    print(f"grad_v_y - Max: {np.max(np.abs(grad_v_y)):.2e}, Mean: {np.mean(np.abs(grad_v_y)):.2e}")
    print(f"grad_w_z - Max: {np.max(np.abs(grad_w_z)):.2e}, Mean: {np.mean(np.abs(grad_w_z)):.2e}")




    # Hydrostatic residual: ∂p'/∂y + g * rho' ≈ 0
    hydrostatic_res = grad_p_y + g * rho_pert
    max_hydro_res = np.max(np.abs(hydrostatic_res))
    mean_hydro_res = np.mean(np.abs(hydrostatic_res))
    print(f"rho_pert - Max: {np.max(np.abs(rho_pert)):.2e}, Mean: {np.max(np.abs(rho_pert)):.2e}")
    print(f"Hydrostatic residual - Max: {max_hydro_res:.2e}, Mean: {mean_hydro_res:.2e}")

    # Geostrophic residuals (for x-z plane, cor in y: f k × v = - (1/rho) ∇p)
    # Assuming u in x, w in z, v in y=0 pert
    # Residual for u: ∂p/∂x + f * rho * w ≈ 0
    # Residual for w: ∂p/∂z - f * rho * u ≈ 0
    geo_res_u = grad_p_x + cor * rho_total * w_total
    geo_res_w = grad_p_z - cor * rho_total * u_total
    max_geo_res = max(np.max(np.abs(geo_res_u)), np.max(np.abs(geo_res_w)))
    mean_geo_res = (np.mean(np.abs(geo_res_u)) + np.mean(np.abs(geo_res_w))) / 2
    print(f"Geostrophic residual - Max: {max_geo_res:.2e}, Mean: {mean_geo_res:.2e}")

    # Thermal wind residual (QG approximation): ∂u/∂y ≈ - (g / (f Y0)) ∂Y/∂z
    # Similarly for ∂w/∂y ≈ (g / (f Y0)) ∂Y/∂x
    therm_wind_u = grad_u_y - (g / (cor * Y0)) * grad_Y_z
    therm_wind_w = grad_w_y + (g / (cor * Y0)) * grad_Y_x
    max_therm_res = max(np.max(np.abs(therm_wind_u)), np.max(np.abs(therm_wind_w)))
    mean_therm_res = (np.mean(np.abs(therm_wind_u)) + np.mean(np.abs(therm_wind_w))) / 2
    print(f"Thermal wind residual - Max: {max_therm_res:.2e}, Mean: {mean_therm_res:.2e}")
    print(f"grad_u_y - Max: {np.max(np.abs(grad_u_y)):.2e}, Mean: {np.max(np.abs(grad_u_y)):.2e}")
    print(f"grad_w_y - Max: {np.max(np.abs(grad_w_y)):.2e}, Mean: {np.max(np.abs(grad_w_y)):.2e}")




    # Field summaries
    print("\nField Summaries:")
    print(f"rho_pert - Min: {np.min(rho_pert):.4f}, Max: {np.max(rho_pert):.4f}, Mean: {np.mean(rho_pert):.4f}")
    print(f"Y_pert - Min: {np.min(Y_pert):.4f}, Max: {np.max(Y_pert):.4f}, Mean: {np.mean(Y_pert):.4f}")
    print(f"Velocity magnitudes - U max: {np.max(np.abs(u_total)):.4f}, W max: {np.max(np.abs(w_total)):.4f}")
    print(f"Rossby number estimate: {np.max(np.abs(u_total)) / (cor * self.R0):.4f} (should be <<1 for QG)")

    print("Diagnostics complete.")


