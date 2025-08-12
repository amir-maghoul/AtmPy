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


def traveling_vortex_stratification(z: float) -> float:
    """
    Isothermal stratification for the Traveling Vortex case.
    Returns a constant value (typically 1.0 in non-dimensional setups).
    The argument corresponds to the vertical coordinate.
    """
    return 1.0


class TravelingVortex3D(BaseTestCase):
    """
    3D Traveling Vortex test case based on the setup described in PyBella and
    the provided C code. This involves an isentropic vortex, uniform along the
    y-axis (vertical), embedded in a uniform flow on a doubly periodic domain (xz) with
    wall boundaries in y. Gravity is set to zero.
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

        super().__init__(name="TravelingVortex3D", config=_effective_config)

        ############################### Vortex Specific Parameters #####################################################
        self.correct_distribution = True

        self.u0: float = 1.0  # Background velocity U
        self.v0: float = 0.0  # Background velocity V
        self.w0: float = 1.0  # Background velocity W
        self.p0: float = 1.0  # Background pressure (dimensionless)

        if self.correct_distribution:
            self.rho0: float = 0.5  # Background density (dimensionless)
            self.del_rho: float = 0.5  # Density deficit at center
            self.alpha = 1
            self.alpha_const = 1
        else:
            self.rho0: float = 1  # Background density (dimensionless)
            self.del_rho: float = -0.5  # Density deficit at center
            self.alpha: float = -1
            self.alpha_const: float = 3

        self.rotdir: float = 1.0  # Rotation direction
        self.R0: float = 0.4  # Vortex radius scale
        self.fac: float = 1.0 * 1024.0  # Vortex magnitude factor
        self.xc: float = 0.0  # Vortex center x
        self.yc: float = 0.0  # Vortex center y
        self.zc: float = 0.0  # Vortex center z

        ####################### Polynomial coefficients for pressure perturbation ######################################
        # Coefficients remain the same as they are based on a 2D radial profile.
        self.coe_correct = np.zeros((25,))
        self.coe_correct[0] = 1.0 / 12.0
        self.coe_correct[1] = -12.0 / 13.0
        self.coe_correct[2] = 9.0 / 2.0
        self.coe_correct[3] = -184.0 / 15.0
        self.coe_correct[4] = 609.0 / 32.0
        self.coe_correct[5] = -222.0 / 17.0
        self.coe_correct[6] = -38.0 / 9.0
        self.coe_correct[7] = 54.0 / 19.0
        self.coe_correct[8] = 783.0 / 20.0
        self.coe_correct[9] = -558.0 / 7.0
        self.coe_correct[10] = 1053.0 / 22.0
        self.coe_correct[11] = 1014.0 / 23.0
        self.coe_correct[12] = -1473.0 / 16.0
        self.coe_correct[13] = 204.0 / 5.0
        self.coe_correct[14] = 510.0 / 13.0
        self.coe_correct[15] = -1564.0 / 27.0
        self.coe_correct[16] = 153.0 / 8.0
        self.coe_correct[17] = 450.0 / 29.0
        self.coe_correct[18] = -269.0 / 15.0
        self.coe_correct[19] = 174.0 / 31.0
        self.coe_correct[20] = 57.0 / 32.0
        self.coe_correct[21] = -74.0 / 33.0
        self.coe_correct[22] = 15.0 / 17.0
        self.coe_correct[23] = -6.0 / 35.0
        self.coe_correct[24] = 1.0 / 72.0

        self.const_coe_correct = np.zeros((13,))
        self.const_coe_correct[0] = 1.0 / 12.0
        self.const_coe_correct[1] = -12.0 / 13.0
        self.const_coe_correct[2] = 33.0 / 7.0
        self.const_coe_correct[3] = -44.0 / 3.0
        self.const_coe_correct[4] = 495.0 / 16.0
        self.const_coe_correct[5] = -792.0 / 17.0
        self.const_coe_correct[6] = 154.0 / 3.0
        self.const_coe_correct[7] = -792.0 / 19.0
        self.const_coe_correct[8] = 99.0 / 4.0
        self.const_coe_correct[9] = -220.0 / 21.0
        self.const_coe_correct[10] = 3.0
        self.const_coe_correct[11] = -12.0 / 23.0
        self.const_coe_correct[12] = 1.0 / 24.0

        self.coe_cor_correct = np.zeros((19,))
        self.coe_cor_correct[0] = 1.0 / 7.0
        self.coe_cor_correct[1] = -3.0 / 4.0
        self.coe_cor_correct[2] = 4.0 / 3.0
        self.coe_cor_correct[3] = -1.0 / 5.0
        self.coe_cor_correct[4] = -45.0 / 22.0
        self.coe_cor_correct[5] = 3.0 / 4.0
        self.coe_cor_correct[6] = 9.0 / 2.0
        self.coe_cor_correct[7] = -36.0 / 7.0
        self.coe_cor_correct[8] = -11.0 / 5.0
        self.coe_cor_correct[9] = 55.0 / 8.0
        self.coe_cor_correct[10] = -33.0 / 17.0
        self.coe_cor_correct[11] = -4.0
        self.coe_cor_correct[12] = 58.0 / 19.0
        self.coe_cor_correct[13] = 3.0 / 5.0
        self.coe_cor_correct[14] = -10.0 / 7.0
        self.coe_cor_correct[15] = 4.0 / 11.0
        self.coe_cor_correct[16] = 9.0 / 46.0
        self.coe_cor_correct[17] = -1.0 / 8.0
        self.coe_cor_correct[18] = 1.0 / 50.0

        self.coe_d_correct = np.zeros((25,))
        self.coe_d_correct[0] = 1.0 / 12.0
        self.coe_d_correct[1] = -12.0 / 13.0
        self.coe_d_correct[2] = 30.0 / 7.0
        self.coe_d_correct[3] = -148.0 / 15.0
        self.coe_d_correct[4] = 57.0 / 8.0
        self.coe_d_correct[5] = 348.0 / 17.0
        self.coe_d_correct[6] = -538.0 / 9.0
        self.coe_d_correct[7] = 900.0 / 19.0
        self.coe_d_correct[8] = 1071.0 / 20.0
        self.coe_d_correct[9] = -3128.0 / 21.0
        self.coe_d_correct[10] = 1020.0 / 11.0
        self.coe_d_correct[11] = 2040.0 / 23.0
        self.coe_d_correct[12] = -1105.0 / 6.0
        self.coe_d_correct[13] = 408.0 / 5.0
        self.coe_d_correct[14] = 1020.0 / 13.0
        self.coe_d_correct[15] = -3128.0 / 27.0
        self.coe_d_correct[16] = 153.0 / 4.0
        self.coe_d_correct[17] = 900.0 / 29.0
        self.coe_d_correct[18] = -538.0 / 15.0
        self.coe_d_correct[19] = 348.0 / 31.0
        self.coe_d_correct[20] = 57.0 / 16.0
        self.coe_d_correct[21] = -148.0 / 33.0
        self.coe_d_correct[22] = 30.0 / 17.0
        self.coe_d_correct[23] = -12.0 / 35.0
        self.coe_d_correct[24] = 1.0 / 36.0

        # Assign coefficients based on the chosen distribution
        if self.correct_distribution:
            self.coe = self.coe_correct
            self.const_coe = self.const_coe_correct
            self.coe_d = self.coe_d_correct
            self.coe_cor = self.coe_cor_correct
        else:
            # Note: C code has specific arrays for incorrect distribution, which are not fully
            # translated here. This part assumes a simplified incorrect case.
            self.coe = self.coe_d_correct  # Example assignment
            self.const_coe = self.const_coe_correct  # Example assignment

        if run_setup_method:
            self.setup()

    def setup(self):
        """Configure the SimulationConfig for the 3D Traveling Vortex case."""
        print("Setting up 3D Traveling Vortex configuration...")

        #################################### Grid Configuration ########################################################
        nx = ny = 64
        nz = 64

        grid_updates = {
            "ndim": 3,
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "xmin": -0.5,
            "xmax": 0.5,
            "ymin": -0.5,
            "ymax": 0.5,
            "zmin": -0.5,
            "zmax": 0.5,
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
            "h_ref": 10_000.0,
            "t_ref": 100.0,
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
            "CFL": 0.45,
            "dtfixed": 0.001,
            "dtfixed0": None,
            "tout": np.array([0.0, 1.0]),
            "tmax": 1.0,
            "stepmax": 10_000,
            "use_acoustic_cfl": True,
        }
        self.set_temporal(temporal_updates)

        #################################### Model Regimes #############################################################
        regime_updates = {
            "is_nongeostrophic": 1,
            "is_nonhydrostatic": 1,
            "is_compressible": 1,
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
        physics_updates = {
            "wind_speed": [self.u0, self.v0, self.w0],
            "gravity_strength": (0.0, 0.0, 0.0), # Gravity is zero for this test case
            "coriolis_strength": (0.0, 0.0, 0.0),
            "stratification": traveling_vortex_stratification,
        }
        self.set_physics(physics_updates)

        #################################### Outputs ###################################################################
        output_updates = {
            "output_type": "test",
            "output_folder": "traveling_vortex_3d",
            "output_base_name": "_traveling_vortex_3d",
            "output_timesteps": True,
            "output_frequency_steps": 20,
        }
        self.set_outputs(output_updates)

        #################################### Diagnostics ###############################################################
        diag_updates = {
            "diag": True,
            "diag_current_run": "atmpy_travelling_vortex_3d",
        }
        self.set_diagnostics(diag_updates)

        ################################################################################################################
        self._update_Msq()
        self._update_output_suffix()

        print(f"Configuration complete. Msq = {self.config.model_regimes.Msq}")
        print(
            f"Output files: {self.config.outputs.output_base_name}{self.config.outputs.output_suffix}"
        )

    def initialize_solution(self, variables: "Variables", mpv: "MPV"):
        """Initialize density, momentum, potential temperature, and pressure fields for 3D."""
        print("Initializing solution for 3D Traveling Vortex...")

        grid = self.config.spatial_grid.grid
        thermo = Thermodynamics()
        thermo.update(self.config.global_constants.gamma)
        Msq = self.config.model_regimes.Msq
        coriolis = self.config.physics.coriolis_strength[1] # Use Fy for consistency with Y-vertical

        # --- Calculate Hydrostatic Base State ---
        gravity = self.config.physics.gravity_strength
        g_axis = self.config.physics.gravity.axis
        mpv.state(gravity, Msq)

        # Get slice for the full inner domain
        inner_slice = grid.get_inner_slice()

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

        # --- Calculate Distance from Vortex Center in the XZ-Plane (Handling Periodicity) ---
        Lx = grid.x_end - grid.x_start
        Lz = grid.z_end - grid.z_start

        dx = XC - self.xc
        dz = ZC - self.zc

        # Account for periodicity
        dx = dx - Lx * np.round(dx / Lx)
        dz = dz - Lz * np.round(dz / Lz)

        # The vortex is uniform in y, so radius is calculated only in the xz-plane
        r_cell = np.sqrt(dx**2 + dz**2)
        r_over_R0_cell = np.divide(r_cell, self.R0, where=self.R0 != 0)

        # --- Calculate Tangential Velocity ---
        uth_cell = np.zeros_like(r_cell)
        mask_cell = (r_cell < self.R0) & (r_cell > 1e-9)
        uth_cell[mask_cell] = (
            self.rotdir
            * self.fac
            * (1.0 - r_over_R0_cell[mask_cell]) ** 6
            * r_over_R0_cell[mask_cell] ** 6
        )

        # --- Calculate Velocity Components (Perturbations in U and W) ---
        u_pert = np.zeros_like(uth_cell)
        w_pert = np.zeros_like(uth_cell)
        u_pert[mask_cell] = uth_cell[mask_cell] * (-dz[mask_cell] / r_cell[mask_cell])
        w_pert[mask_cell] = uth_cell[mask_cell] * (+dx[mask_cell] / r_cell[mask_cell])

        u_total = self.u0 + u_pert
        w_total = self.w0 + w_pert
        v_total = np.full_like(u_total, self.v0) # v-velocity is uniform background

        # --- Calculate Density ---
        rho_total = np.full_like(r_cell, self.rho0)
        mask_rho = r_cell < self.R0
        rho_total[mask_rho] += self.del_rho * (1.0 - r_over_R0_cell[mask_rho] ** 2) ** 6

        # --- Calculate Pressure Perturbation (dp2c) ---
        dp2c = np.zeros_like(r_cell)
        if Msq > 1e-10:
            # Main pressure term
            term_main = np.zeros_like(r_cell)
            for ip, c in enumerate(self.coe):
                term = self.fac**2 * c * (r_over_R0_cell ** (12 + ip) - 1.0) * self.rotdir**2
                term_main[mask_rho] += term[mask_rho]

            # Coriolis/Constant term
            dp2c_const = np.zeros_like(r_cell)
            if self.correct_distribution:
                for ip, c in enumerate(self.coe_cor):
                    term = self.fac * coriolis * c * (r_over_R0_cell ** (7 + ip) - 1.0) * self.R0
                    dp2c_const[mask_rho] += term[mask_rho]
            else:
                 for ip, c in enumerate(self.const_coe):
                    term = self.fac**2 * c * (r_over_R0_cell ** (12 + ip) - 1.0) * self.rotdir**2
                    dp2c_const[mask_rho] += term[mask_rho]

            dp2c = self.alpha * term_main + self.alpha_const * dp2c_const
        else:
            print("Msq is near zero, pressure perturbation dp2c set to zero.")

        # --- Assign to Cell Variables (Inner Domain Only) ---
        variables.cell_vars[inner_slice + (VI.RHO,)] = rho_total
        variables.cell_vars[inner_slice + (VI.RHOU,)] = rho_total * u_total
        variables.cell_vars[inner_slice + (VI.RHOV,)] = rho_total * v_total
        variables.cell_vars[inner_slice + (VI.RHOW,)] = rho_total * w_total

        # --- Handle rhoY (Potential Temperature * Density) ---
        # Get ghost cell info
        ng_g_axis = self.config.spatial_grid.grid.ng[g_axis][0]

        # Extract the 1D hydrostatic variable (it includes ghost cells)
        rhoY0_cells_1d_full = mpv.hydrostate.cell_vars[..., HI.RHOY0]

        # Slice out the inner part of the 1D array to remove ghost cells
        inner_slice_1d = slice(ng_g_axis, -ng_g_axis if ng_g_axis > 0 else None)
        rhoY0_cells_1d_inner = rhoY0_cells_1d_full[inner_slice_1d]

        # Reshape for broadcasting based on the gravity axis
        reshape_dims = [1] * grid.ndim
        reshape_dims[g_axis] = -1  # e.g., (1, -1, 1) for g_axis=1
        rhoY0_reshaped = rhoY0_cells_1d_inner.reshape(tuple(reshape_dims))

        # Broadcast to the full 3D inner domain shape
        target_shape_3d = variables.cell_vars[inner_slice + (VI.RHO,)].shape
        rhoY0_cells = np.broadcast_to(rhoY0_reshaped, target_shape_3d)

        if self.config.model_regimes.is_compressible:
            p_total = self.p0 + Msq * dp2c
            p_total_safe = np.maximum(p_total, 1e-9)
            variables.cell_vars[inner_slice + (VI.RHOY,)] = p_total_safe**thermo.gamminv
        else:
            variables.cell_vars[inner_slice + (VI.RHOY,)] = rhoY0_cells

        if VI.RHOX < variables.num_vars_cell:
            variables.cell_vars[inner_slice + (VI.RHOX,)] = 0.0

        # --- Calculate Nodal Pressure Perturbation p2 (Nodes, Inner Domain Only) ---
        if grid.ndim == 3:
             XN, YN, ZN = np.meshgrid(
                grid.x_nodes[inner_slice[0]],
                grid.y_nodes[inner_slice[1]],
                grid.z_nodes[inner_slice[2]],
                indexing="ij",
            )
        else:
            raise NotImplementedError("Nodal calculation requires 3D grid")

        dx_node = XN - self.xc
        dz_node = ZN - self.zc
        dx_node = dx_node - Lx * np.round(dx_node / Lx)
        dz_node = dz_node - Lz * np.round(dz_node / Lz)

        r_node = np.sqrt(dx_node**2 + dz_node**2) # Radius is in xz-plane
        r_over_R0_node = np.divide(r_node, self.R0, where=self.R0 != 0)
        mask_node = r_node < self.R0

        p2_nodes_unscaled = np.zeros_like(r_node)

        # Main term
        term_main_node = np.zeros_like(r_node)
        for ip, c in enumerate(self.coe):
            term = self.fac**2 * c * (r_over_R0_node ** (12 + ip) - 1.0) * self.rotdir**2
            term_main_node[mask_node] += term[mask_node]

        # Coriolis/Constant term
        p2n_const_unscaled = np.zeros_like(r_node)
        if self.correct_distribution:
            for ip, c in enumerate(self.coe_cor):
                term = self.fac * coriolis * c * (r_over_R0_node ** (7 + ip) - 1.0) * self.R0
                p2n_const_unscaled[mask_node] += term[mask_node]
        else:
            for ip, c in enumerate(self.const_coe):
                term = self.fac**2 * c * (r_over_R0_node ** (12 + ip) - 1.0) * self.rotdir**2
                p2n_const_unscaled[mask_node] += term[mask_node]

        p2_nodes_unscaled = self.alpha * term_main_node + self.alpha_const * p2n_const_unscaled

        # --- Correctly broadcast the nodal hydrostatic state ---
        rhoY0_nodes_1d_full = mpv.hydrostate.node_vars[..., HI.RHOY0]
        rhoY0_nodes_1d_inner = rhoY0_nodes_1d_full[inner_slice_1d]

        reshape_dims_nodes = [1] * grid.ndim
        reshape_dims_nodes[g_axis] = -1
        rhoY0_nodes_reshaped = rhoY0_nodes_1d_inner.reshape(tuple(reshape_dims_nodes))

        target_shape_3d_nodes = mpv.p2_nodes[inner_slice].shape
        rhoY0_nodes = np.broadcast_to(rhoY0_nodes_reshaped, target_shape_3d_nodes)

        mpv.p2_nodes[inner_slice] = thermo.Gamma * np.divide(
            p2_nodes_unscaled, rhoY0_nodes, where=rhoY0_nodes != 0
        )

        logging.info("3D solution initialization complete.")