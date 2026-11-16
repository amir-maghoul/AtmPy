"""Steady, pressure-generated thermal-wind test for atmpy.

Coordinates: x is along the flow; y is vertical; z is across the flow.
Only y and z are active spatial coordinates, but all velocity components and
Coriolis terms are retained by using the 3D framework with a thin x grid.

Unscaled Exner anomaly:
    delta_pi(y,z) = A * sin(2*pi*z/Lz) * B(y/H),
    B(eta) = (1-eta**2)**3 for |eta|<1, and 0 otherwise.
The nodal pressure variable is pi2 = delta_pi / Msq.

Geostrophic/hydrostatic construction with chi=1/theta, S0=1/theta0:
    chi_prime = -Msq/(Gamma*g) * D_y(pi2)
    theta = 1/(S0 + chi_prime)
    u_geo = theta/(f*Gamma) * D_z(pi2)
    v = w = 0
    rho = P/theta, RHOX = rho*chi_prime
Compressible: P=(pi0 + delta_pi_cells)**(1/(gamma-1)).
Pseudo-incompressible/QG: P=P0.

All fields are independent of x and the velocity points along x. Consequently,
there is no advective acceleration or transport of the initial fields. This is
an equilibrium-maintenance test, not an instability or wave-propagation test.
The force balance is enforced using the supplied nodal-to-cell gradient kernel.
Time splitting, boundary handling and numerical diffusion can still cause drift.

Physical defaults: f=1e-4 /s, N=1e-2 /s, 3 m/s peak geostrophic speed,
2000 km cross-flow wavelength, 12 km vertical domain, 5 km anomaly half-depth.
The origin is at the midlevel: physical altitude = y*h_ref + 6000 m.
The uniform background wind is 10 m/s along x and is passed through wind_speed.
All modes use the SAME dt_seconds (default 0.25 s) and CFL policy.

Requires the existing atmpy package. No solver modifications are made here.
"""

import logging
from typing import TYPE_CHECKING

import numpy as np

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
    VariableIndices as VI,
    HydrostateIndices as HI,
    SlopeLimiters as LimiterType,
)
from atmpy.physics.thermodynamics import Thermodynamics
from atmpy.pressure_solver.operators_numba_kernels import _gradient_3d_numba

if TYPE_CHECKING:
    from atmpy.variables.variables import Variables
    from atmpy.variables.multiple_pressure_variables import MPV


class ThermalWind3D(BaseTestCase):
    def __init__(
        self,
        config_override: SimulationConfig = None,
        *,
        mode: str = "qg",
        target_shear_wind_ms: float = 3.0,
        dt_seconds: float = 0.25,
        duration_hours: float = 1.0,
        resolution=(5, 33, 65),
    ):
        # ----------------------------------------------------------------------
        # Input validation and model modes
        # ----------------------------------------------------------------------
        regimes = {
            "qg": (0, 0, 0),
            "pseudo_incompressible": (1, 1, 0),
            "compressible": (1, 1, 1),
        }
        if mode not in regimes:
            raise ValueError("mode must be qg, pseudo_incompressible, or compressible")
        if not all(np.isfinite(v) and v > 0 for v in (
            target_shear_wind_ms, dt_seconds, duration_hours
        )):
            raise ValueError("Wind amplitude, timestep and duration must be positive")
        if (len(resolution) != 3 or any(int(n) != n for n in resolution)
                or resolution[0] < 3 or min(resolution[1:]) < 9):
            raise ValueError("Require integer grid counts: nx>=3, ny>=9, nz>=9")

        config = config_override if config_override is not None else SimulationConfig()
        super().__init__(name="ThermalWind3D", config=config)

        # ----------------------------------------------------------------------
        # Reference scales and dimensional physical parameters
        # ----------------------------------------------------------------------
        self.h_ref = 10_000.0
        self.t_ref = 100.0
        self.velocity_ref = self.h_ref / self.t_ref
        self.gamma = 1.4
        self.R_gas = 287.4
        self.T_ref = 300.0
        self.p_ref = 100_000.0
        self.g_ms2 = 9.81
        self.f_s = 1.0e-4
        self.N_s = 1.0e-2
        self.Nsq_ref = self.N_s**2
        self.coriolis_strength = self.f_s * self.t_ref
        self.g = None  # Assigned once the framework has computed Msq.

        # ----------------------------------------------------------------------
        # Background wind (nondimensional): 10 m/s along x
        # ----------------------------------------------------------------------
        self.u0 = 0.1
        self.v0 = 0.0
        self.w0 = 0.0

        # ----------------------------------------------------------------------
        # Geometry and pressure amplitude
        # ----------------------------------------------------------------------
        self.x_length_m = 200_000.0
        self.z_length_m = 2_000_000.0
        self.height_m = 12_000.0
        self.altitude_offset_m = 0.5 * self.height_m
        self.half_depth_m = 5_000.0
        self.yc = 0.0
        self.zc = 0.0
        self.H_scale = self.half_depth_m / self.h_ref
        self.target_shear_wind_ms = float(target_shear_wind_ms)
        self.p_amp = None  # Calibrated to the requested speed at initialization.
        self.resolution = tuple(int(n) for n in resolution)

        # ----------------------------------------------------------------------
        # Model switches
        # ----------------------------------------------------------------------
        self.mode = mode
        (
            self.is_nongeostrophic,
            self.is_nonhydrostatic,
            self.is_compressible,
        ) = regimes[mode]
        self.initial_projection = False

        # ----------------------------------------------------------------------
        # Common time controls for every regime
        # ----------------------------------------------------------------------
        self.dt_seconds = float(dt_seconds)
        self.duration_hours = float(duration_hours)
        self.dtfixed = self.dt_seconds / self.t_ref
        self.dtfixed0 = None
        self.tmax = self.duration_hours * 3600.0 / self.t_ref
        self.stepmax = int(np.ceil(self.tmax / self.dtfixed)) + 10
        self.acoustic_cfl = False
        self.output_freq = max(1, round(300.0 / self.dt_seconds))
        self.checkpoint_freq = max(1, round(3600.0 / self.dt_seconds))

        if config_override is None:
            self.setup()

    # --------------------------------------------------------------------------
    # Simulation configuration
    # --------------------------------------------------------------------------
    def setup(self):
        # Grid: periodic x/z, walls in y; center at y=0.
        nx, ny, nz = self.resolution
        grid_updates = {
            "ndim": 3,
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "xmin": -0.5 * self.x_length_m / self.h_ref,
            "xmax": +0.5 * self.x_length_m / self.h_ref,
            "ymin": -0.5 * self.height_m / self.h_ref,
            "ymax": +0.5 * self.height_m / self.h_ref,
            "zmin": -0.5 * self.z_length_m / self.h_ref,
            "zmax": +0.5 * self.z_length_m / self.h_ref,
            "ngx": 2,
            "ngy": 2,
            "ngz": 2,
        }
        self.set_grid_configuration(grid_updates)

        # Global constants.
        constants_updates = {
            "gamma": self.gamma,
            "R_gas": self.R_gas,
            "p_ref": self.p_ref,
            "T_ref": self.T_ref,
            "h_ref": self.h_ref,
            "t_ref": self.t_ref,
            "Nsq_ref": self.Nsq_ref,
            "grav": self.g_ms2,
        }
        self.set_global_constants(constants_updates)

        # Model regimes.
        regime_updates = {
            "is_nongeostrophic": self.is_nongeostrophic,
            "is_nonhydrostatic": self.is_nonhydrostatic,
            "is_compressible": self.is_compressible,
        }
        self.set_model_regimes(regime_updates)
        self._update_Msq()

        # Physics. Match g/Msq to dimensional gravity in the momentum equation.
        Msq = self.config.model_regimes.Msq
        self.g = Msq * self.g_ms2 * self.t_ref**2 / self.h_ref
        physics_updates = {
            "wind_speed": [self.u0, self.v0, self.w0],
            "gravity_strength": (0.0, self.g, 0.0),
            "coriolis_strength": (0.0, self.coriolis_strength, 0.0),
            "stratification": self.thermal_wind_stratification,
        }
        self.set_physics(physics_updates)

        # Boundary conditions.
        for side in (
            BoundarySide.LEFT, BoundarySide.RIGHT,
            BoundarySide.FRONT, BoundarySide.BACK,
        ):
            self.set_boundary_condition(
                side, BdryType.PERIODIC, mpv_type=BdryType.PERIODIC
            )
        for side in (BoundarySide.BOTTOM, BoundarySide.TOP):
            self.set_boundary_condition(
                side, BdryType.REFLECTIVE_GRAVITY, mpv_type=BdryType.WALL
            )
        self.config.boundary_conditions.sponge.active = False

        # Time integration: identical settings in all modes.
        temporal_updates = {
            "CFL": 0.45,
            "dtfixed": self.dtfixed,
            "dtfixed0": self.dtfixed0,
            "tout": np.append(np.arange(0.0, self.tmax, 300.0 / self.t_ref), self.tmax),
            "tmax": self.tmax,
            "stepmax": self.stepmax,
            "use_acoustic_cfl": self.acoustic_cfl,
        }
        self.set_temporal(temporal_updates)

        # Numerical methods.
        numerics_updates = {
            "limiter": LimiterType.AVERAGE,
            "riemann_solver": RiemannSolvers.MODIFIED_HLL,
            "reconstruction": FluxReconstructions.MODIFIED_MUSCL,
            "first_order_advection_routine": AdvectionRoutines.FIRST_ORDER_RK,
            "second_order_advection_routine": AdvectionRoutines.STRANG_SPLIT,
            "linear_solver": LinearSolvers.BICGSTAB,
            "preconditioner": Preconditioners.DIAGONAL,
            "initial_projection": self.initial_projection,
        }
        self.set_numerics(numerics_updates)

        # Outputs and checkpoints.
        output_updates = {
            "output_type": "test",
            "output_folder": "thermal_wind_3d",
            "output_base_name": "_thermal_wind_3d",
            "checkpoint_base_name": "_thermal_wind_3d_checkpoint",
            "output_timesteps": True,
            "output_frequency_steps": self.output_freq,
            "checkpoint_frequency_steps": self.checkpoint_freq,
        }
        self.set_outputs(output_updates)

        # Diagnostics.
        diag_updates = {
            "diag": True,
            "diag_current_run": "atmpy_thermal_wind_3d",
        }
        self.set_diagnostics(diag_updates)
        self._update_output_suffix()

    # --------------------------------------------------------------------------
    # Reference state and generating pressure function
    # --------------------------------------------------------------------------
    def thermal_wind_stratification(self, y):
        """Constant-N atmosphere; y=0 corresponds to 6 km physical altitude."""
        height_m = np.asarray(y) * self.h_ref + self.altitude_offset_m
        return np.exp(self.N_s**2 * height_m / self.g_ms2)

    def _pressure_shape(self, Y, Z):
        """Unit unscaled Exner anomaly; independent of x and periodic in z."""
        grid = self.config.spatial_grid.grid
        Lz = grid.z_end - grid.z_start
        eta = (Y - self.yc) / self.H_scale
        inside = np.abs(eta) < 1.0
        vertical = np.zeros_like(Y)
        vertical[inside] = (1.0 - eta[inside]**2)**3
        return np.sin(2.0 * np.pi * (Z - self.zc) / Lz) * vertical

    @staticmethod
    def _broadcast_y(profile, shape):
        profile = np.asarray(profile)
        if profile.ndim != 1 or profile.size != shape[1]:
            raise ValueError("Hydrostate profile does not match the vertical grid")
        return np.broadcast_to(profile[None, :, None], shape)

    @staticmethod
    def _nodes_to_cells(p):
        return 0.125 * (
            p[:-1, :-1, :-1] + p[1:, :-1, :-1]
            + p[:-1, 1:, :-1] + p[:-1, :-1, 1:]
            + p[1:, 1:, :-1] + p[1:, :-1, 1:]
            + p[:-1, 1:, 1:] + p[1:, 1:, 1:]
        )

    # --------------------------------------------------------------------------
    # Initial conditions
    # --------------------------------------------------------------------------
    def initialize_solution(self, variables: "Variables", mpv: "MPV"):
        # 1. Validate geometry, units and physical configuration.
        grid = self.config.spatial_grid.grid
        gl = self.config.global_constants
        if grid.ndim != 3 or self.config.physics.gravity.axis != 1:
            raise ValueError("ThermalWind3D requires three dimensions with y vertical")
        if not np.isclose(gl.h_ref, self.h_ref) or not np.isclose(gl.t_ref, self.t_ref):
            raise ValueError("Reference scales must be h_ref=10000 m and t_ref=100 s")
        Msq = float(self.config.model_regimes.Msq)
        if not np.isfinite(Msq) or Msq <= 0:
            raise ValueError("Positive finite Msq is required in all regimes")
        gravity = np.asarray(self.config.physics.gravity_strength, dtype=float)
        rotation = np.asarray(self.config.physics.coriolis_strength, dtype=float)
        wind = np.asarray(self.config.physics.wind_speed, dtype=float)
        g, f = gravity[1], rotation[1]
        if np.any(gravity[[0, 2]] != 0) or g <= 0:
            raise ValueError("Gravity must be positive along the y-axis")
        if np.any(rotation[[0, 2]] != 0) or f == 0:
            raise ValueError("Coriolis must be nonzero along the y-axis")
        if not np.isclose(f / self.t_ref, self.f_s):
            raise ValueError("Coriolis does not match the physical test parameters")
        if not np.isclose(g / Msq * self.h_ref / self.t_ref**2, self.g_ms2):
            raise ValueError("Gravity does not represent 9.81 m/s^2")
        if wind.shape != (3,) or wind[1] != 0 or wind[2] != 0:
            raise ValueError("A steady test allows background wind only along x")
        if (self.yc - self.H_scale <= grid.y_start
                or self.yc + self.H_scale >= grid.y_end):
            raise ValueError("Vertical pressure support must fit inside the walls")

        # 2. Use the framework's own hydrostatic reference state.
        thermo = Thermodynamics()
        thermo.update(gl.gamma)
        Gamma = thermo.Gamma
        compressible = bool(self.config.model_regimes.is_compressible)
        mpv.state(tuple(gravity), Msq)
        cshape = variables.cell_vars.shape[:-1]
        nshape = mpv.p2_nodes.shape
        if tuple(n - 1 for n in nshape) != cshape:
            raise ValueError("Expected one more node than cell along every axis")
        P0 = self._broadcast_y(mpv.hydrostate.cell_vars[..., HI.RHOY0], cshape)
        S0 = self._broadcast_y(mpv.hydrostate.cell_vars[..., HI.S0], cshape)
        P0n = self._broadcast_y(mpv.hydrostate.node_vars[..., HI.RHOY0], nshape)
        if np.any(P0 <= 0) or np.any(P0n <= 0) or np.any(S0 <= 0):
            raise ValueError("Reference thermodynamic fields must be positive")
        pi0 = P0**(thermo.gamma - 1.0)
        pi0n = P0n**(thermo.gamma - 1.0)
        inner = grid.get_inner_slice()

        # 3. Build the nodal shape and use the actual pressure-gradient stencil.
        YN, ZN = np.meshgrid(grid.y_nodes, grid.z_nodes, indexing="ij")
        shape = np.broadcast_to(self._pressure_shape(YN, ZN)[None, :, :], nshape).copy()
        dx, dy, dz = _gradient_3d_numba(shape, *grid.dxyz)

        # 4. Set the pressure amplitude from the requested geostrophic speed.
        # With A>0: chi=S0-A*D, speed=A*C/(S0-A*D).
        # Solve the target-speed equation in each cell and take the smallest
        # positive amplitude; the fastest cell then reaches the target exactly.
        D = dy / (Gamma * g)
        C = self.velocity_ref * np.abs(dz) / (abs(f) * Gamma * Msq)
        denominator = C[inner] + self.target_shear_wind_ms * D[inner]
        valid = (denominator > 0) & (C[inner] > 0)
        if not np.any(valid):
            raise ValueError("The grid does not resolve a nonzero thermal-wind flow")
        amplitude = np.min(
            self.target_shear_wind_ms * S0[inner][valid] / denominator[valid]
        )
        self.p_amp = float(amplitude)
        chi_prime = -self.p_amp * D
        chi = S0 + chi_prime
        if np.any(chi <= 0) or not np.all(np.isfinite(chi)):
            raise ValueError("Requested amplitude produces non-positive theta")
        theta = 1.0 / chi

        # 5. Regime-specific thermodynamic pressure.
        physical_anomaly = self.p_amp * shape
        pi2 = physical_anomaly / Msq
        gx, gy, gz = (self.p_amp / Msq * a for a in (dx, dy, dz))
        physical_cells = self._nodes_to_cells(physical_anomaly)
        if np.any(pi0 + physical_cells <= 0) or np.any(pi0n + physical_anomaly <= 0):
            raise ValueError("Total Exner pressure must remain positive")
        if compressible:
            P = P0 * np.exp(
                np.log1p(physical_cells / pi0) / (thermo.gamma - 1.0)
            )
        else:
            P = P0.copy()
        rho = P * chi

        # 6. Geostrophic velocity: flow strictly along x.
        u_geo = theta * gz / (f * Gamma)
        heights_m = np.asarray(grid.y_cells) * self.h_ref + self.altitude_offset_m
        N2_total = self.g_ms2 * np.gradient(
            np.log(theta), heights_m, axis=1, edge_order=2
        )
        if np.min(N2_total[inner]) <= 0:
            raise ValueError("Pressure anomaly overturns stratification; reduce target speed")

        # 7. Assign cell variables and nodal pressure, including ghost coordinates.
        variables.cell_vars[..., VI.RHO] = rho
        variables.cell_vars[..., VI.RHOU] = rho * (wind[0] + u_geo)
        variables.cell_vars[..., VI.RHOV] = 0.0
        variables.cell_vars[..., VI.RHOW] = 0.0
        variables.cell_vars[..., VI.RHOY] = P
        variables.cell_vars[..., VI.RHOX] = rho * chi_prime
        mpv.p2_nodes[...] = pi2

        # 8. Initial force-balance checks and reference data for drift monitoring.
        force_x = (P / Gamma) * gx
        force_y = (P / Gamma) * gy + (g / Msq) * P * chi_prime
        force_z = (P / Gamma) * gz - f * rho * u_geo
        peak_speed = float(np.max(np.abs(u_geo[inner])) * self.velocity_ref)
        Lz_m = (grid.z_end - grid.z_start) * self.h_ref
        self.initial_diagnostics = {
            "f_physical_s-1": f / self.t_ref,
            "f_code": f,
            "peak_shear_flow_m_s-1": peak_speed,
            "background_flow_m_s-1": float(wind[0] * self.velocity_ref),
            "Rossby_number_based_on_inverse_wavenumber": (
                peak_speed * (2.0 * np.pi / Lz_m) / abs(self.f_s)
            ),
            "physical_Exner_amplitude": self.p_amp,
            "minimum_N2_s-2": float(np.min(N2_total[inner])),
            "max_theta_anomaly_K": float(
                np.max(np.abs((theta - 1.0 / S0)[inner])) * gl.T_ref
            ),
            "max_force_residual_x": float(np.max(np.abs(force_x[inner]))),
            "max_force_residual_y": float(np.max(np.abs(force_y[inner]))),
            "max_force_residual_z": float(np.max(np.abs(force_z[inner]))),
        }
        self._initial_cells = variables.cell_vars[inner].copy()
        self._initial_pi2 = mpv.p2_nodes[inner].copy()
        logging.info("Thermal-wind initialization: %s", self.initial_diagnostics)

    # --------------------------------------------------------------------------
    # Optional diagnostic: call after a timestep or at an output time
    # --------------------------------------------------------------------------
    def measure_drift(self, variables: "Variables", mpv: "MPV"):
        """Compare current interior fields with the stationary initialized state.

        This method does not run automatically and does not modify the state.
        Pressure shape error removes a spatially uniform offset, which can be
        arbitrary in the soundproof regimes; the mean shift is also reported.
        """
        if not hasattr(self, "_initial_cells"):
            raise RuntimeError("initialize_solution must be called first")
        inner = self.config.spatial_grid.grid.get_inner_slice()
        current = variables.cell_vars[inner]
        initial = self._initial_cells
        pressure = mpv.p2_nodes[inner]
        reference = self._initial_pi2
        rho = current[..., VI.RHO]
        rho_initial = initial[..., VI.RHO]
        u = current[..., VI.RHOU] / rho
        u_initial = initial[..., VI.RHOU] / rho_initial
        theta = current[..., VI.RHOY] / rho
        theta_initial = initial[..., VI.RHOY] / rho_initial
        difference = pressure - reference
        shape_difference = difference - np.mean(difference)
        reference_shape = reference - np.mean(reference)
        return {
            "max_relative_density_change": float(
                np.max(np.abs(rho - rho_initial) / rho_initial)
            ),
            "max_u_change_m_s-1": float(np.max(np.abs(u - u_initial)) * self.velocity_ref),
            "max_vertical_speed_m_s-1": float(
                np.max(np.abs(current[..., VI.RHOV] / rho)) * self.velocity_ref
            ),
            "max_cross_flow_speed_m_s-1": float(
                np.max(np.abs(current[..., VI.RHOW] / rho)) * self.velocity_ref
            ),
            "max_theta_change_K": float(
                np.max(np.abs(theta - theta_initial)) * self.config.global_constants.T_ref
            ),
            "relative_pi2_shape_L2_error": float(
                np.linalg.norm(shape_difference.ravel())
                / max(np.linalg.norm(reference_shape.ravel()), np.finfo(float).tiny)
            ),
            "pi2_mean_shift": float(np.mean(difference)),
        }
