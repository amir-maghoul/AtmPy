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

def _gradient_3d(
    p: np.ndarray, dx: float, dy: float, dz: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numba kernel for 3D gradient (cell-centered).

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

        self.u0: float = 0.0  # Background velocity U
        self.v0: float = 0.0  # Background velocity V
        self.w0: float = 0.0  # Background velocity W
        self.p0: float = 1.0  # Background pressure (dimensionless)

        if self.correct_distribution:
            self.rho0: float = 0.5  # Background density (dimensionless)
            self.del_rho: float = 0.5  # Density deficit at center
            self.alpha = 0
            self.alpha_const = 1

        self.xc: float = 0.0  # Vortex center x
        self.yc: float = 0.0  # Vortex center y
        self.zc: float = 0.0  # Vortex center z

        self.instability = False
        self.jet_amplitude: float = 0.5  # Max speed of the jet
        self.jet_width: float = 0.2     # How sharp the jet is
        self.rotdir: float = 1.0  # Rotation direction
        self.R0: float = 0.5  # Vortex radius scale
        self.fac: float = 1.0 if self.instability else 1.0  # Vortex magnitude factor

        self.baroclinic = False
        self.stratified_atmosphere = False
        self.constant_rho = False
        self.eps = 1.0e-3
        self.vortex_vertical_scale: float = 0.5
        self.dtfixed0 = 0.0001
        self.dtfixed = self.dtfixed0 if self.dtfixed0 else 0.1
        self.tmax = 1.0
        self.acoustic_cfl = True

        self.output_freq = 1
        self.boxsize = 0.5
        self.h_ref = 10_000.0
        self.t_ref = 100.0
        self.g = 1.0 if self.stratified_atmosphere else 0.0
        coriolis_factor = 100000
        force = 1.0e-4 * self.t_ref * coriolis_factor  # Constant coriolis force
        self.cor_f = [0.0, 0.0, 0.0]
        self.coriolis_axis = 1
        self.cor_f[self.coriolis_axis] = force

        self.is_nongeostrophic = 0
        self.is_nonhydrostatic = 1
        self.is_compressible = 1

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

        self.coe_cor_constant_rho = np.zeros((7,))
        self.coe_cor_constant_rho[0] = 1.0 / 7.0
        self.coe_cor_constant_rho[1] = -3.0 / 4.0
        self.coe_cor_constant_rho[2] = 5.0 / 3.0
        self.coe_cor_constant_rho[3] = -2.0
        self.coe_cor_constant_rho[4] = -15.0 / 11.0
        self.coe_cor_constant_rho[5] = -1.0 / 2.0
        self.coe_cor_constant_rho[6] = 1.0 / 13.0

        # Assign coefficients based on the chosen distribution
        if self.correct_distribution:
            self.coe = self.coe_correct
            self.const_coe = self.const_coe_correct
            self.coe_d = self.coe_d_correct
            self.coe_cor = self.coe_cor_correct if not self.constant_rho else self.coe_cor_constant_rho
        else:
            self.coe = self.coe_d_correct  # Example assignment
            self.const_coe = self.const_coe_correct  # Example assignment

        if run_setup_method:
            self.setup()

    def setup(self):
        """Configure the SimulationConfig for the 3D Traveling Vortex case."""
        print("Setting up 3D Traveling Vortex configuration...")

        #################################### Grid Configuration ########################################################
        nx = nz = 40
        ny = 3

        boxsize = self.boxsize # This should be always less than 1.0 otherwise the stratification creates inf due to
                      # large dx.

        grid_updates = {
            "ndim": 3,
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "xmin": -boxsize,
            "xmax": boxsize,
            "ymin": -boxsize,
            "ymax": boxsize,
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
            "Nsq_ref": 1.0e-4,
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
            "stratification": self.qg_stratification if self.baroclinic else traveling_vortex_stratification,
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
        return np.exp(Nsq_scaled * y / g_eff)

    def jet_stream(self, z: np.ndarray):
        """ Defining jet stream for barotropic instability

        Parameters
        ----------
        z : np.ndarray
            The second component of the horizontal cells array
        """
        u_background = self.jet_amplitude * np.tanh(z / self.jet_width)
        w_background = np.zeros_like(u_background) # No background meridional flow

        return u_background, w_background

    def vertical_decay_function(self, y:np.ndarray, rho):
        """ Vertical decay of hydrostatic perturbation"""

        return self.eps * rho[:, np.newaxis] * np.exp(-(y ** 2) / self.vortex_vertical_scale ** 2)

    def initialize_solution(self, variables: "Variables", mpv: "MPV"):
        """Initialize density, momentum, potential temperature, and pressure fields for 3D.

        Notice the initial states are:

        rho(x, y, z) = rho_c * 1/2 * (1-r^2)^6 if r <= 1 else rho_c [notice rho_c is 1/2 above]
        u(x, y, z) = -1024 * sin(Θ) * (1-r)^6 * r^6 + u_c if r <= 1 else u_c
        v(x, y, z) = 1024 * cos(Θ) * (1-r)^6 * r^6 + v_c if r <= 1 else v_c

        therefore the tangential velocity is:
        V_Θ = 1024 * (1-r)^6 * r^6

        Radial momentum balance equation:
        dp/dr = rho * (V_theta ^2 / r + f V_theta) with f is the CONSTANT coriolis force in the vertical direction.

        ----------------------------------------------------------------------------------------------------------------
        from Wolfram|Alpha:

         Radial pressure gradient for constant density
          (1-x)^12 * x^11 =
                x^11 - 12 x^12 + 66 x^13 - 220 x^14 + 495 x^15 - 792 x^16 + 924 x^17 - 792 x^18 + 495 x^19 - 220 x^20
                + 66 x^21 - 12 x^22 + x^23

         Radial pressure distribution as deviation from pressure at the center for constant density
          int_0^x ((1-y)^12 * y^11) dy
                x^12/12 - (12 x^13)/13 + (33 x^14)/7 - (44 x^15)/3 + (495 x^16)/16 - (792 x^17)/17 + (154 x^18)/3
                -(792 x^19)/19 + (99 x^20)/4 - (220 x^21)/21 + 3 x^22 - (12 x^23)/23 + x^24/24

         Radial pressure gradient for the density variation
          (1-x^2)^6 * (1-x)^12 * x^11 =
                x^35 - 12 x^34 + 60 x^33 - 148 x^32 + 114 x^31 + 348 x^30 - 1076 x^29 + 900 x^28 + 1071 x^27
                - 3128 x^26 + 2040 x^25 + 2040 x^24 - 4420 x^23 + 2040 x^22 + 2040 x^21 - 3128 x^20 + 1071 x^19
                + 900 x^18 - 1076 x^17 + 348 x^16 + 114 x^15 - 148 x^14 + 60 x^13 - 12 x^12 + x^11

         Radial pressure distribution as deviation from pressure at the center for the variable density part
         int_0^x ((1-y^2)^6 * (1-y)^12 * y^11) dy =
                x^12/12 - (12 x^13)/13 + (30 x^14)/7 - (148 x^15)/15 + (57 x^16)/8 + (348 x^17)/17 - (538 x^18)/9 +
                (900 x^19)/19 + (1071 x^20)/20 - (3128 x^21)/21 + (1020 x^22)/11 + (2040 x^23)/23 - (1105 x^24)/6 +
                (408 x^25)/5 + (1020 x^26)/13 - (3128 x^27)/27 + (153 x^28)/4 + (900 x^29)/29 - (538 x^30)/15 +
                (348 x^31)/31 + (57 x^32)/16 - (148 x^33)/33 + (30 x^34)/17 - (12 x^35)/35 + x^36/36



        """
        print("Initializing solution for 3D Traveling Vortex...")

        grid = self.config.spatial_grid.grid
        thermo = Thermodynamics()
        thermo.update(self.config.global_constants.gamma)
        Msq = self.config.model_regimes.Msq

        # --- Calculate Hydrostatic Base State ---
        gravity = self.config.physics.gravity_strength
        g_axis = self.config.physics.gravity.axis
        mpv.state(gravity, Msq)
        coriolis = self.config.physics.coriolis_strength[
            self.coriolis_axis
        ]  # Coriolis force in z-direction which is *NOT* the vertical direction

        # Get slice for the full inner domain
        inner_slice = grid.get_inner_slice()
        inner_slice_1d = inner_slice[g_axis]

        # --- Calculate broadcasting of 1D stratification profile to 3D --- #
        # rho0_1d_inner = mpv.hydrostate.cell_vars[inner_slice_1d, HI.RHO0]
        # rhoY0_1d_inner = mpv.hydrostate.cell_vars[inner_slice_1d, HI.RHOY0]
        # Y0_1d_inner = rhoY0_1d_inner / rho0_1d_inner
        #
        # # Reshape for broadcasting based on the gravity axis
        # reshape_dims = [1] * grid.ndim
        # reshape_dims[g_axis] = -1  # e.g., (1, -1, 1) for g_axis=1
        # target_shape = variables.cell_vars[inner_slice + (VI.RHO,)].shape
        # rhoY0_reshaped = rhoY0_1d_inner.reshape(tuple(reshape_dims))
        # rhoY0_3d_inner = np.broadcast_to(rhoY0_reshaped, target_shape)
        # rho0_3d_inner = np.broadcast_to(rho0_1d_inner.reshape(tuple(reshape_dims)), target_shape)
        # Y0_3d_inner = np.broadcast_to(Y0_1d_inner.reshape(tuple(reshape_dims)), target_shape)

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

        # The vortex is uniform in y, so the radius is calculated only on the xz-plane
        r_cell = np.sqrt(dx**2 + dz**2)
        r_over_R0_cell = np.divide(r_cell, self.R0, where=self.R0 != 0)

        # 0. Define density field
        rho_pert = np.zeros_like(r_cell)
        mask_rho = r_cell < self.R0
        rho_total = np.full_like(r_cell, self.rho0)
        rho_pert[mask_rho] = self.del_rho * (1.0 - r_over_R0_cell[mask_rho] ** 2) ** 6
        rho_total[mask_rho] += rho_pert[mask_rho]


        # 1. Define pressure anomaly
        # p_amplitude = -0.05  # A negative value for a low-pressure center (cyclone)
        # vortex_radius = self.R0
        # # This pressure field is the "slow manifold" or the "riverbanks".
        # dp2c = np.zeros_like(r_cell)
        # dp2c[mask_rho] = p_amplitude * np.exp(-(r_cell[mask_rho]**2) / (vortex_radius**2))
        #
        # # 2. Derive the balanced velocity from pressure field
        # # The analytical derivative of p_amplitude * exp(-r²/R²) is:
        # dp_dr = np.zeros_like(r_cell)
        # dp_dr[mask_rho] = dp2c[mask_rho] * (-2 * r_cell[mask_rho] / (vortex_radius**2))
        #
        # # --- Calculate Tangential Velocity ---
        # uth_cell = np.zeros_like(r_cell)
        # mask_cell = (r_cell < self.R0) & (r_cell > 1e-9)
        # # uth_cell[mask_cell] = (
        # #     self.rotdir
        # #     * self.fac
        # #     * (1.0 - r_over_R0_cell[mask_cell]) ** 6
        # #     * r_over_R0_cell[mask_cell] ** 6
        # # )
        # uth_cell[mask_cell] = (1.0 / (rho_total[mask_cell] * coriolis)) * dp_dr[mask_cell]
        #
        # # --- Calculate Velocity Components (Perturbations in U and W) ---
        # u_pert = np.zeros_like(uth_cell)
        # w_pert = np.zeros_like(uth_cell)
        # u_pert[mask_cell] = uth_cell[mask_cell] * (-dz[mask_cell] / r_cell[mask_cell])
        # w_pert[mask_cell] = uth_cell[mask_cell] * (+dx[mask_cell] / r_cell[mask_cell])
        #
        # if self.instability and not self.baroclinic:
        #     u_background, w_background = self.jet_stream(ZC)
        # else:
        #     u_background, w_background = self.u0, self.w0
        #
        # u_total = u_background + u_pert
        # w_total = w_background + w_pert
        # v_total = np.full_like(u_total, self.v0)  # v-velocity is uniform background
        #
        # mask_rho = r_cell < self.R0

        # ==================================================================================
        # ====== RESTRUCTURED AND CORRECTED PERTURBATION CALCULATION =======================
        # ==================================================================================

        # --- 1. Define Universal Perturbation Fields (Valid for both cases) ---

        # --- Calculate Pressure Perturbation (dp2c) ---
        p_geostrophic_pert = np.zeros_like(r_cell)
        if Msq > 1e-10:
            # Main pressure term
            term_main = np.zeros_like(r_cell)
            for ip, c in enumerate(self.coe):
                term = (
                    self.fac**2
                    * c
                    * (r_over_R0_cell ** (12 + ip) - 1.0)
                    * self.rotdir**2
                )
                term_main[mask_rho] += term[mask_rho]

            # Coriolis/Constant term
            dp2c_const = np.zeros_like(r_cell)
            if self.correct_distribution:
                for ip, c in enumerate(self.coe_cor):
                    term = (
                        self.fac
                        * coriolis
                        * c
                        * (r_over_R0_cell ** (7 + ip) - 1.0)
                        * self.R0
                    )
                    dp2c_const[mask_rho] += term[mask_rho]
            else:
                for ip, c in enumerate(self.const_coe):
                    term = (
                        self.fac**2
                        * c
                        * (r_over_R0_cell ** (12 + ip) - 1.0)
                        * self.rotdir**2
                    )
                    dp2c_const[mask_rho] += term[mask_rho]

            # dp2c_const *= self.vertical_decay_function(YC, rho0_1d_inner) if self.constant_rho else 1.0
            dp2c = self.alpha * term_main + self.alpha_const * dp2c_const
        # else:
        #     print("Msq is near zero, pressure perturbation dp2c set to zero.")

        # Create the vertical decay scale to make the perturbation hydrostatic in case needed.
        # rho_pert = np.zeros_like(r_cell)
        # mask_rho = r_cell < self.R0
        # rho_pert[mask_rho] = self.del_rho * (1.0 - r_over_R0_cell[mask_rho] ** 2) ** 6 \
        #     if not self.constant_rho else self.vertical_decay_function(YC, rho0_1d_inner)[mask_rho]

        # The total dynamic pressure perturbation is the sum of the two balanced components
        # p_dynamic_pert = p_geostrophic_pert

        # --- Calculate Density and rhoY ---
        # if self.stratified_atmosphere:
        #     # The hydrostatic pressure perturbation (vertical balance)
        #     # p_hydrostatic_pert = mpv.integrate_density_hydrostatically(rho_pert, gravity)
        #     # p_dynamic_pert += p_hydrostatic_pert
        #     # p_hydro_3d = rhoY0_3d_inner ** thermo.gamma
        #     # p_total_3d = p_hydro_3d + Msq * p_dynamic_pert
        #     # p_total_3d = np.maximum(p_total_3d, 1e-9)
        #
        #     # Derive total rhoY and then total rho
        #     rho_total = rho0_3d_inner + rho_pert
        #     rhoY_total_3d = rho_total * Y0_3d_inner
        # else:
        #     p_jet_pert = -self.rho0 * self.cor_f[self.coriolis_axis] * self.jet_amplitude \
        #                                * self.jet_width * np.log(np.cosh(ZC / self.jet_width))
        #     p_vortex_pert = p_dynamic_pert
        #     p_dynamic_pert = p_vortex_pert + p_jet_pert if self.instability else p_vortex_pert
        #     rho_total = np.full_like(r_cell, self.rho0)
        #     rho_total[mask_rho] += rho_pert[mask_rho]
        #     if self.config.model_regimes.is_compressible:
        #         p_total = self.p0 + Msq * p_dynamic_pert
        #         p_total_safe = np.maximum(p_total, 1e-9)
        #         rhoY_total_3d = p_total_safe ** thermo.gamminv
        #     else:
        #         rhoY_total_3d = rhoY0_3d_inner

        if self.config.model_regimes.is_compressible:
            p_total = self.p0 + Msq * dp2c
            p_total_safe = np.maximum(p_total, 1e-9)
            rhoY_total_3d = p_total_safe ** thermo.gamminv

        # --- Assign to Cell Variables (Inner Domain Only) ---
        variables.cell_vars[inner_slice + (VI.RHO,)] = rho_total
        # variables.cell_vars[inner_slice + (VI.RHOU,)] = rho_total * u_total
        # variables.cell_vars[inner_slice + (VI.RHOV,)] = rho_total * v_total
        # variables.cell_vars[inner_slice + (VI.RHOW,)] = rho_total * w_total
        variables.cell_vars[inner_slice + (VI.RHOY,)] = rhoY_total_3d
        variables.cell_vars[inner_slice + (VI.RHOX,)] = 0.0


########################################################################################################################
######### --- Calculate Nodal Pressure Perturbation p2 (Nodes, Inner Domain Only) --- ##################################
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

        r_node = np.sqrt(dx_node**2 + dz_node**2)  # Radius is in xz-plane
        r_over_R0_node = np.divide(r_node, self.R0, where=self.R0 != 0)
        mask_node = r_node < self.R0

        # Main term
        term_main_node = np.zeros_like(r_node)
        for ip, c in enumerate(self.coe):
            term = (
                self.fac**2 * c * (r_over_R0_node ** (12 + ip) - 1.0) * self.rotdir**2
            )
            term_main_node[mask_node] += term[mask_node]

        # Coriolis/Constant term
        p2n_const_unscaled = np.zeros_like(r_node)
        if self.correct_distribution:
            for ip, c in enumerate(self.coe_cor):
                term = (
                    self.fac
                    * coriolis
                    * c
                    * (r_over_R0_node ** (7 + ip) - 1.0)
                    * self.R0
                )
                p2n_const_unscaled[mask_node] += term[mask_node]
        else:
            for ip, c in enumerate(self.const_coe):
                term = (
                    self.fac**2
                    * c
                    * (r_over_R0_node ** (12 + ip) - 1.0)
                    * self.rotdir**2
                )
                p2n_const_unscaled[mask_node] += term[mask_node]

        # p2n_const_unscaled *= self.vertical_decay_function(YN, rho0_1d_inner) if self.constant_rho else 1.0
        dp2c_n = (
            self.alpha * term_main_node + self.alpha_const * p2n_const_unscaled
        )

        rho_pert = np.zeros_like(r_node)
        mask_rho = r_node < self.R0
        rho_pert[mask_rho] = self.del_rho * (1.0 - r_over_R0_node[mask_rho] ** 2) ** 6

        # The total dynamic pressure perturbation is the sum of the two balanced components
        # p_dynamic_pert_n = p_geostrophic_pert_n
        #
        # if self.stratified_atmosphere:
        #     # The hydrostatic pressure perturbation (vertical balance)
        #     p_hydrostatic_pert_n = mpv.integrate_density_hydrostatically(rho_pert, gravity)
        #     p_dynamic_pert_n += p_hydrostatic_pert_n
        #     rhoY0_nodes_1d_inner = mpv.hydrostate.node_vars[inner_slice_1d, HI.RHOY0]
        #     rhoY0_nodes_reshaped = rhoY0_nodes_1d_inner.reshape(tuple(reshape_dims))
        #     target_shape = mpv.p2_nodes[inner_slice].shape
        #     rhoY0_3d_nodes = np.broadcast_to(rhoY0_nodes_reshaped, target_shape)
        #
        #     # p_hydro_nodes = rhoY0_3d_nodes ** thermo.gamma
        #     # p_total_nodes = p_hydro_nodes + Msq * p_dynamic_pert_n
        #     # p_total_nodes = np.maximum(p_total_nodes, 1e-9)
        #     # rhoY_total_nodes = p_total_nodes ** (1.0 / thermo.gamma)
        #     #
        #     # # The relationship is π = P^(gamma-1)
        #     # pi_total_nodes = thermo.Gamma * (rhoY_total_nodes ** thermo.gm1)
        #     # pi_hydro_nodes = thermo.Gamma * (rhoY0_3d_nodes ** thermo.gm1)
        #     # pi_pert_nodes = pi_total_nodes - pi_hydro_nodes
        #
        #     mpv.p2_nodes[inner_slice] = thermo.Gamma * np.divide(
        #         p_dynamic_pert_n, rhoY0_3d_nodes, where=rhoY0_3d_nodes != 0
        #     )
        #
        #     # mpv.p2_nodes[inner_slice] = pi_pert_nodes
        # else:
        #     p_jet_pert = -self.rho0 * self.cor_f[self.coriolis_axis] * self.jet_amplitude \
        #                                * self.jet_width * np.log(np.cosh(ZN / self.jet_width))
        #     p_vortex_pert = p_dynamic_pert_n
        #     p_dynamic_pert = p_vortex_pert + p_jet_pert if self.instability else p_vortex_pert
        #     background_p = self.p0
        #     p_total_nodes = background_p + Msq * p_dynamic_pert_n
        #     rhoY0_nodes = p_total_nodes**thermo.gamminv
        #
        #     mpv.p2_nodes[inner_slice] = thermo.Gamma * np.divide(
        #         p_dynamic_pert_n, rhoY0_nodes, where=rhoY0_nodes != 0
        #     )

        # # --- Correctly broadcast the nodal hydrostatic state ---
        rhoY0_nodes_1d_full = mpv.hydrostate.node_vars[..., HI.RHOY0]
        rhoY0_nodes_1d_inner = rhoY0_nodes_1d_full[inner_slice_1d]

        reshape_dims_nodes = [1] * grid.ndim
        reshape_dims_nodes[g_axis] = -1
        rhoY0_nodes_reshaped = rhoY0_nodes_1d_inner.reshape(tuple(reshape_dims_nodes))

        target_shape_3d_nodes = mpv.p2_nodes[inner_slice].shape
        rhoY0_nodes = np.broadcast_to(rhoY0_nodes_reshaped, target_shape_3d_nodes)


        mpv.p2_nodes[inner_slice] = thermo.Gamma * np.divide(
                    dp2c_n, rhoY0_nodes, where=rhoY0_nodes != 0
                )

        Dpx, Dpy, Dpz = _gradient_3d(mpv.p2_nodes[inner_slice], grid.dx, grid.dy, grid.dz)
        Theta_c = np.divide(rhoY_total_3d, rho_total, where=rho_total != 0.0)
        u_c = - (Theta_c / (thermo.Gamma * coriolis)) * Dpz
        w_c = + (Theta_c / (thermo.Gamma * coriolis)) * Dpx

        variables.cell_vars[inner_slice + (VI.RHOU,)] = rho_total * u_c
        variables.cell_vars[inner_slice + (VI.RHOV,)] = 0.0  # unchanged in this 2D-xz vortex
        variables.cell_vars[inner_slice + (VI.RHOW,)] = rho_total * w_c

        p = np.zeros_like(r_node)
        R0 = 0.2
        p[mask_rho] = np.exp(-(dx_node[mask_rho]**2 + dz_node[mask_rho]**2) / (2 * R0**2))
        print("khar")
        mpv.p2_nodes[inner_slice] = p

        logging.info("3D solution initialization complete.")
