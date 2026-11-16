"""Compact baroclinic vortex and a resolved internal-gravity-wave seed.

Coordinates: (x, y, z); y is vertical and v is vertical velocity.
Defaults: 800 x 12 x 800 km, (48, 16, 48), 90 minutes, 10-second steps.
Use anomaly_kind="packet", resolution=(64, 16, 48) for horizontal packets.

The wave period (~88 minutes) is a local linear estimate. The grid and time
step are demonstration choices, not a verified convergence minimum.
See baroclinic_qg_vortex3D_parameters.md for every parameter and equation.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple
import logging
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


@dataclass
class _InitialState:
    """Temporary arrays used while constructing the initial condition.

    rho_theta is the original P = rho * theta variable. All arrays are in
    code units. The inverse-theta anomaly is 1/theta - 1/theta_background.
    """

    rho: np.ndarray
    rho_theta: np.ndarray
    theta: np.ndarray
    inverse_theta_anomaly: np.ndarray
    reference_inverse_theta: np.ndarray
    u: np.ndarray
    w: np.ndarray
    pressure_nodes: np.ndarray
    pressure_gradients: Tuple[np.ndarray, np.ndarray, np.ndarray]
    nodes_x: np.ndarray
    nodes_y: np.ndarray


@dataclass(frozen=True)
class _WaveParameters:
    """Dimensional wave scales and actual grid samples per wavelength."""

    horizontal_wavelength_m: float
    vertical_wavelength_m: float
    points_per_shortest_horizontal_wave: float
    points_per_vertical_wave: float
    horizontal_wavenumber: float
    vertical_wavenumber: float
    angular_frequency: float

    @property
    def period_seconds(self):
        return 2.0 * np.pi / self.angular_frequency


class BaroclinicQGVortex(BaseTestCase):
    """Geostrophic/hydrostatic vortex with an optional unbalanced thermal seed.

    The public constructor and framework hooks match the supplied test case.
    With config_override supplied, the caller owns setup; initialization
    checks that the configured regime, units and domain agree with this case.
    """

    # These are numerical policies, not constants derived from wave theory.
    GHOST_CELLS = 2
    CFL = 0.45
    MIN_GRID_COUNT = 9
    MIN_DOMAIN_RADII = 6.0
    MIN_HORIZONTAL_WAVE_POINTS = 10
    MIN_VERTICAL_WAVE_POINTS = 12
    CHECKPOINT_INTERVAL_SECONDS = 1800.0
    MIN_STEP_LIMIT = 10_000
    STEP_LIMIT_HEADROOM = 10

    def __init__(
            self,
            config_override: SimulationConfig = None,
            *,
            mode: str = "qg",
            target_wind_ms: float = 0.6,
            f_s: float = 1.0e-4,
            N_s: float = 1.0e-2,
            duration_hours: float = 1.5,
            resolution=(48, 16, 48),
            initial_projection: bool = False,
            add_unbalanced_anomaly: bool = False,
            thermal_anomaly_K: float = 0.5,
            anomaly_kind: str = "mode",
            anomaly_horizontal_mode: int = 4,
            anomaly_vertical_mode: int = 1,
            radius_m: float = 100_000.0,
            half_depth_m: float = 5_000.0,
            center_height_m: float = 6_000.0,
            domain_width_m: float = 800_000.0,
            domain_height_m: float = 12_000.0,
            dt_seconds: float = None,
            output_interval_seconds: float = 60.0,
    ):
        regimes = {
            "qg": (0, 0, 0),
            "pseudo_incompressible": (1, 1, 0),
            "compressible": (1, 1, 1),
        }
        if mode not in regimes:
            raise ValueError(f"mode must be one of {tuple(regimes)}")
        if anomaly_kind not in ("mode", "packet"):
            raise ValueError("anomaly_kind must be 'mode' or 'packet'")
        positive = (target_wind_ms, N_s, duration_hours, radius_m,
                    half_depth_m, center_height_m, domain_width_m,
                    domain_height_m, output_interval_seconds)
        if not all(np.isfinite(a) and a > 0 for a in positive):
            raise ValueError("Wind, N, times and length scales must be finite and positive")
        if not np.isfinite(f_s) or f_s == 0:
            raise ValueError("f_s must be finite and nonzero")
        if not np.isfinite(thermal_anomaly_K):
            raise ValueError("thermal_anomaly_K must be finite")
        invalid_counts = any(
            not np.isfinite(n) or n != int(n) or n < self.MIN_GRID_COUNT
            for n in resolution
        )
        if len(resolution) != 3 or invalid_counts:
            raise ValueError("resolution requires three integer grid counts >= 9")
        for n in (anomaly_horizontal_mode, anomaly_vertical_mode):
            if not np.isfinite(n) or n != int(n) or n < 1:
                raise ValueError("Anomaly mode numbers must be positive integers")
        if anomaly_kind == "packet" and anomaly_horizontal_mode < 2:
            raise ValueError("A zero-mean packet requires horizontal mode >= 2")
        if not (0 < center_height_m - half_depth_m and
                center_height_m + half_depth_m < domain_height_m):
            raise ValueError("The vortex must fit strictly between the vertical walls")
        if domain_width_m < self.MIN_DOMAIN_RADII * radius_m:
            raise ValueError("Use domain_width_m >= 6 * radius_m for an isolated vortex")

        effective = config_override if config_override is not None else SimulationConfig()
        super().__init__(name="BaroclinicQGVortex", config=effective)
        self.mode = mode
        self.add_unbalanced_anomaly = bool(add_unbalanced_anomaly)
        self.thermal_anomaly_K = float(thermal_anomaly_K)
        self.anomaly_kind = anomaly_kind
        self.anomaly_horizontal_mode = int(anomaly_horizontal_mode)
        self.anomaly_vertical_mode = int(anomaly_vertical_mode)
        self.target_wind_ms = float(target_wind_ms)
        self.f_s = float(f_s)
        self.N_s = float(N_s)
        self.g_ms2 = 9.81
        self.radius_m = float(radius_m)
        self.half_depth_m = float(half_depth_m)
        self.center_height_m = float(center_height_m)
        self.domain_width_m = float(domain_width_m)
        self.domain_height_m = float(domain_height_m)
        self.resolution = tuple(int(n) for n in resolution)

        # Preserve the original reference units and thermodynamic conventions.
        self.h_ref = 10_000.0
        self.t_ref = 100.0
        self.R_gas = 287.4
        self.T_ref = 300.0
        self.p_ref = 100_000.0
        self.gamma = 1.4
        self.velocity_ref = self.h_ref / self.t_ref
        self.Nsq_ref = self.N_s ** 2
        self.coriolis_strength = self.f_s * self.t_ref
        self.g = None
        self.u0 = self.v0 = self.w0 = 0.0
        self.xc = self.zc = 0.0
        self.yc = self.center_height_m / self.h_ref
        self.R0 = self.radius_m / self.h_ref
        self.H_scale = self.half_depth_m / self.h_ref
        self.p_amp = None
        self.baroclinic = self.stratified_atmosphere = True
        (self.is_nongeostrophic, self.is_nonhydrostatic,
         self.is_compressible) = regimes[mode]
        self.initial_projection = bool(initial_projection)

        # Small acoustic steps are needed only for the optional compressible run.
        if dt_seconds is None:
            dt_seconds = 0.2 if self.is_compressible else 10.0
        if not np.isfinite(dt_seconds) or dt_seconds <= 0:
            raise ValueError("dt_seconds must be finite and positive")
        self.dt_seconds = float(dt_seconds)
        self.duration_hours = float(duration_hours)
        self.tmax = self.duration_hours * 3600.0 / self.t_ref
        self.dtfixed = self.dt_seconds / self.t_ref
        self.dtfixed0 = None
        # Generous headroom if the driver shortens a step to satisfy CFL.
        self.stepmax = max(self.MIN_STEP_LIMIT, self.STEP_LIMIT_HEADROOM
                           * int(np.ceil(self.tmax / self.dtfixed)))
        self.output_interval_seconds = float(output_interval_seconds)
        self.output_freq = max(1, int(round(self.output_interval_seconds / self.dt_seconds)))
        self.checkpoint_freq = max(
            1, int(round(self.CHECKPOINT_INTERVAL_SECONDS / self.dt_seconds))
        )

        if self.mode == "qg" and self.add_unbalanced_anomaly and self.thermal_anomaly_K:
            logging.warning("QG mode filters gravity waves; use pseudo_incompressible to see them")
        if config_override is None:
            self.setup()

    def setup(self):
        """Apply the case's settings through the original atmpy API."""
        self._configure_grid()
        self._configure_physics()
        self._configure_boundaries()
        self._configure_time()
        self._configure_numerics()
        self._configure_outputs()
        self._update_output_suffix()

    def _configure_grid(self):
        half_width = 0.5 * self.domain_width_m / self.h_ref
        nx, ny, nz = self.resolution
        self.set_grid_configuration({
            "ndim": 3,
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "xmin": -half_width,
            "xmax": half_width,
            "ymin": 0.0,
            "ymax": self.domain_height_m / self.h_ref,
            "zmin": -half_width,
            "zmax": half_width,
            "ngx": self.GHOST_CELLS,
            "ngy": self.GHOST_CELLS,
            "ngz": self.GHOST_CELLS,
        })

    def _configure_physics(self):
        self.set_global_constants({
            "gamma": self.gamma,
            "R_gas": self.R_gas,
            "p_ref": self.p_ref,
            "T_ref": self.T_ref,
            "h_ref": self.h_ref,
            "t_ref": self.t_ref,
            "Nsq_ref": self.Nsq_ref,
            "grav": self.g_ms2,
        })
        self.set_model_regimes({
            "is_nongeostrophic": self.is_nongeostrophic,
            "is_nonhydrostatic": self.is_nonhydrostatic,
            "is_compressible": self.is_compressible,
        })
        self._update_Msq()
        Msq = self.config.model_regimes.Msq
        self.g = Msq * self.g_ms2 * self.t_ref ** 2 / self.h_ref
        self.set_physics({
            "wind_speed": [self.u0, self.v0, self.w0],
            "gravity_strength": (0.0, self.g, 0.0),
            "coriolis_strength": (0.0, self.coriolis_strength, 0.0),
            "stratification": self.qg_stratification,
        })

    def _configure_boundaries(self):
        horizontal_sides = (
            BoundarySide.LEFT, BoundarySide.RIGHT,
            BoundarySide.FRONT, BoundarySide.BACK,
        )
        for side in horizontal_sides:
            self.set_boundary_condition(
                side, BdryType.PERIODIC, mpv_type=BdryType.PERIODIC
            )
        for side in (BoundarySide.BOTTOM, BoundarySide.TOP):
            self.set_boundary_condition(
                side, BdryType.REFLECTIVE_GRAVITY, mpv_type=BdryType.WALL
            )
        self.config.boundary_conditions.sponge.active = False

    def _configure_time(self):
        output_interval = self.output_interval_seconds / self.t_ref
        output_times = np.append(
            np.arange(0.0, self.tmax, output_interval), self.tmax
        )
        self.set_temporal({
            "CFL": self.CFL,
            "dtfixed": self.dtfixed,
            "dtfixed0": self.dtfixed0,
            "tout": output_times,
            "tmax": self.tmax,
            "stepmax": self.stepmax,
            "use_acoustic_cfl": bool(self.is_compressible),
        })

    def _configure_numerics(self):
        # Retained from the supplied case; these are solver choices.
        self.set_numerics({
            "limiter": LimiterType.AVERAGE,
            "riemann_solver": RiemannSolvers.MODIFIED_HLL,
            "reconstruction": FluxReconstructions.MODIFIED_MUSCL,
            "first_order_advection_routine": AdvectionRoutines.FIRST_ORDER_RK,
            "second_order_advection_routine": AdvectionRoutines.STRANG_SPLIT,
            "linear_solver": LinearSolvers.BICGSTAB,
            "preconditioner": Preconditioners.DIAGONAL,
            "initial_projection": self.initial_projection,
        })

    def _configure_outputs(self):
        state = "on" if self._anomaly_is_active() else "off"
        run_name = f"baroclinic_qg_vortex_{state}"
        self.set_outputs({
            "output_type": "test",
            "output_folder": run_name,
            "output_base_name": "_" + run_name,
            "checkpoint_base_name": "_" + run_name + "_checkpoint",
            "output_timesteps": True,
            "output_frequency_steps": self.output_freq,
            "checkpoint_frequency_steps": self.checkpoint_freq,
        })
        self.set_diagnostics({"diag": True, "diag_current_run": run_name})

    def _anomaly_is_active(self):
        return self.add_unbalanced_anomaly and self.thermal_anomaly_K != 0.0

    def qg_stratification(self, y):
        """Dimensionless theta0; g * d(log(theta0))/d(height) = N**2."""
        return np.exp(self.N_s ** 2 * np.asarray(y) * self.h_ref / self.g_ms2)

    def _gaussian_perturbation(self, X, Y, Z, H_scale):
        """Original compact-depth pressure vortex, with nearest periodic images."""
        grid = self.config.spatial_grid.grid
        Lx = grid.x_end - grid.x_start
        Lz = grid.z_end - grid.z_start
        dx = X - self.xc
        dz = Z - self.zc
        dx = dx - Lx * np.round(dx / Lx)
        dz = dz - Lz * np.round(dz / Lz)
        horizontal = np.exp(-(dx ** 2 + dz ** 2) / (2.0 * self.R0 ** 2))
        eta = (Y - self.yc) / H_scale
        inside = np.abs(eta) < 1.0
        vertical = np.zeros_like(Y)
        d_vertical_dy = np.zeros_like(Y)
        term = 1.0 - eta[inside] ** 2
        vertical[inside] = term ** 3
        d_vertical_dy[inside] = -6.0 * eta[inside] * term ** 2 / H_scale
        anomaly = self.p_amp * horizontal * vertical
        return (anomaly, -dx * anomaly / self.R0 ** 2,
                self.p_amp * horizontal * d_vertical_dy, -dz * anomaly / self.R0 ** 2)

    def _thermal_wave_nodes(self, X, Y):
        """Zero-horizontal-mean wave seed; deliberately uniform in horizontal z.

        'mode':   cos(2*pi*n*x/Lx) * sin(pi*j*y/D).
        'packet': multiply by cos(pi*x/Lx)**2, adding sidebands n-1,n+1.
        Both are smooth and exactly periodic in x; n>=2 keeps the packet
        free of a horizontally uniform thermal component. y is measured
        from the lower wall. The cell-averaged field is normalized later.
        """
        grid = self.config.spatial_grid.grid
        Lx = grid.x_end - grid.x_start
        depth = grid.y_end - grid.y_start
        phase = 2.0 * np.pi * (X - self.xc) / Lx
        horizontal = np.cos(self.anomaly_horizontal_mode * phase)
        if self.anomaly_kind == "packet":
            horizontal = horizontal * (0.5 + 0.5 * np.cos(phase))
        vertical = np.sin(self.anomaly_vertical_mode * np.pi * (Y - grid.y_start) / depth)
        return horizontal * vertical

    @staticmethod
    def _nodes_to_cells(values):
        return 0.125 * (
                values[:-1, :-1, :-1] + values[1:, :-1, :-1]
                + values[:-1, 1:, :-1] + values[:-1, :-1, 1:]
                + values[1:, 1:, :-1] + values[1:, :-1, 1:]
                + values[:-1, 1:, 1:] + values[1:, 1:, 1:])

    @staticmethod
    def _broadcast_y(profile, shape):
        profile = np.asarray(profile)
        if profile.ndim != 1 or profile.size != shape[1]:
            raise ValueError("Hydrostate profile must match full y-grid shape")
        return np.broadcast_to(profile[None, :, None], shape)

    def initialize_solution(self, variables: "Variables", mpv: "MPV"):
        """Construct balance, add the wave seed, then assign framework fields."""
        self._validate_configuration()

        state = self._build_balanced_state(variables, mpv)
        wave = self._wave_parameters()
        self._check_wave_resolution(wave)
        self._add_thermal_anomaly(state)
        minimum_N2 = self._check_static_stability(state.theta)

        self._assign_initial_state(variables, mpv, state)
        self._record_diagnostics(state, wave, minimum_N2)

    def _validate_configuration(self):
        """Reject incompatible caller-owned configurations before initialization."""
        grid, gl = self.config.spatial_grid.grid, self.config.global_constants
        if grid.ndim != 3 or self.config.physics.gravity.axis != 1:
            raise ValueError("Require 3D with y vertical")
        if not (np.isclose(gl.h_ref, self.h_ref) and np.isclose(gl.t_ref, self.t_ref)
                and np.isclose(gl.T_ref, self.T_ref)):
            raise ValueError("Use h_ref=10000 m, t_ref=100 s and T_ref=300 K")
        flags = self.config.model_regimes
        actual_regime = (flags.is_nongeostrophic, flags.is_nonhydrostatic, flags.is_compressible)
        if actual_regime != (self.is_nongeostrophic, self.is_nonhydrostatic, self.is_compressible):
            raise ValueError("config_override regime disagrees with mode")
        extents = np.array([grid.x_end - grid.x_start, grid.y_end - grid.y_start,
                            grid.z_end - grid.z_start]) * self.h_ref
        expected_extents = (
            self.domain_width_m, self.domain_height_m, self.domain_width_m
        )
        if not np.allclose(extents, expected_extents):
            raise ValueError("config_override domain disagrees with the constructor geometry")
        gravity = np.asarray(self.config.physics.gravity_strength, dtype=float)
        rotation = np.asarray(self.config.physics.coriolis_strength, dtype=float)
        wind = np.asarray(self.config.physics.wind_speed, dtype=float)
        if gravity.shape != (3,) or rotation.shape != (3,) or wind.shape != (3,):
            raise ValueError("gravity, rotation and wind must have three components")
        if not all(np.all(np.isfinite(a)) for a in (gravity, rotation, wind)):
            raise ValueError("gravity, rotation and wind must be finite")
        Msq, g, f = float(flags.Msq), gravity[1], rotation[1]
        if not np.isfinite(Msq) or Msq <= 0:
            raise ValueError("Positive finite Msq is required in every mode")
        if np.any(gravity[[0, 2]] != 0) or g <= 0:
            raise ValueError("Require positive y gravity and zero horizontal gravity")
        if np.any(rotation[[0, 2]] != 0) or f == 0 or wind[1] != 0:
            raise ValueError("Require nonzero y Coriolis and horizontal background wind")
        g_physical = g / Msq * self.h_ref / self.t_ref ** 2
        if not np.isclose(f / self.t_ref, self.f_s, rtol=1e-8):
            raise ValueError("Configured Coriolis disagrees with f_s")
        if not np.isclose(g_physical, self.g_ms2, rtol=1e-8):
            raise ValueError("Configured gravity does not represent 9.81 m/s^2")
        if self.yc - self.H_scale <= grid.y_start or self.yc + self.H_scale >= grid.y_end:
            raise ValueError("Vertical vortex support must fit strictly inside the walls")

    def _build_balanced_state(self, variables, mpv):
        """Calibrate pressure and enforce the original discrete force balance."""
        grid = self.config.spatial_grid.grid
        gl = self.config.global_constants
        flags = self.config.model_regimes
        gravity = np.asarray(self.config.physics.gravity_strength, dtype=float)
        g = gravity[1]
        f = self.config.physics.coriolis_strength[1]
        Msq = float(flags.Msq)
        th = Thermodynamics()
        th.update(gl.gamma)
        Gamma = th.Gamma
        mpv.state(tuple(gravity), Msq)
        cshape, nshape = variables.cell_vars.shape[:-1], mpv.p2_nodes.shape
        if tuple(n - 1 for n in nshape) != cshape:
            raise ValueError("Expected one more node than cell in every direction")
        P0 = self._broadcast_y(mpv.hydrostate.cell_vars[..., HI.RHOY0], cshape)
        S0 = self._broadcast_y(mpv.hydrostate.cell_vars[..., HI.S0], cshape)
        P0n = self._broadcast_y(mpv.hydrostate.node_vars[..., HI.RHOY0], nshape)
        if any(np.any(a <= 0) or not np.all(np.isfinite(a)) for a in (P0, S0, P0n)):
            raise ValueError("Reference atmosphere must be positive and finite")
        pi0, pi0n = P0 ** (th.gamma - 1), P0n ** (th.gamma - 1)
        inner = grid.get_inner_slice()
        XN, YN, ZN = np.meshgrid(grid.x_nodes, grid.y_nodes, grid.z_nodes, indexing="ij")

        # Calibrate the pressure amplitude against the actual discrete stencil.
        self.p_amp = 1.0
        shape, _, _, _ = self._gaussian_perturbation(XN, YN, ZN, self.H_scale)
        dx, dy, dz = _gradient_3d_numba(shape, *grid.dxyz)
        D = dy / (Gamma * g)
        C = self.velocity_ref * np.hypot(dx, dz) / (abs(f) * Gamma * Msq)
        denominator = C[inner] - self.target_wind_ms * D[inner]
        valid = denominator > 0
        if not np.any(valid):
            raise ValueError("The grid does not resolve the vortex gradient")
        amplitude = np.min(self.target_wind_ms * S0[inner][valid] / denominator[valid])
        self.p_amp = -float(amplitude)
        chi_prime = amplitude * D
        chi = S0 + chi_prime
        if np.any(chi <= 0) or not np.all(np.isfinite(chi)):
            raise ValueError("Requested wind gives invalid potential temperature")
        theta = 1.0 / chi
        physical_anomaly = self.p_amp * shape
        pi2 = physical_anomaly / Msq
        gx, gy, gz = (self.p_amp / Msq * a for a in (dx, dy, dz))
        physical_cells = self._nodes_to_cells(physical_anomaly)
        if np.any(pi0 + physical_cells <= 0) or np.any(pi0n + physical_anomaly <= 0):
            raise ValueError("Requested wind gives non-positive total Exner pressure")
        if bool(flags.is_compressible):
            P = P0 * np.exp(np.log1p(physical_cells / pi0) / (th.gamma - 1))
        else:
            P = P0.copy()
        rho = P * chi
        u_geo, w_geo = theta * gz / (f * Gamma), -theta * gx / (f * Gamma)

        return _InitialState(
            rho=rho,
            rho_theta=P,
            theta=theta,
            inverse_theta_anomaly=chi_prime,
            reference_inverse_theta=S0,
            u=u_geo,
            w=w_geo,
            pressure_nodes=pi2,
            pressure_gradients=(gx, gy, gz),
            nodes_x=XN,
            nodes_y=YN,
        )

    def _wave_parameters(self):
        """Evaluate physical wavelengths, dispersion and grid resolution."""
        grid = self.config.spatial_grid.grid
        width_m = (grid.x_end - grid.x_start) * self.h_ref
        depth_m = (grid.y_end - grid.y_start) * self.h_ref
        dx_m = grid.dxyz[0] * self.h_ref
        dy_m = grid.dxyz[1] * self.h_ref

        horizontal_mode = self.anomaly_horizontal_mode
        vertical_mode = self.anomaly_vertical_mode
        highest_mode = horizontal_mode + int(self.anomaly_kind == "packet")
        wavelength_h = width_m / horizontal_mode
        wavelength_v = 2.0 * depth_m / vertical_mode
        kh = 2.0 * np.pi / wavelength_h
        m = 2.0 * np.pi / wavelength_v
        omega = np.sqrt(
            (self.N_s ** 2 * kh ** 2 + self.f_s ** 2 * m ** 2) / (kh ** 2 + m ** 2)
        )
        return _WaveParameters(
            horizontal_wavelength_m=wavelength_h,
            vertical_wavelength_m=wavelength_v,
            points_per_shortest_horizontal_wave=(width_m / highest_mode) / dx_m,
            points_per_vertical_wave=wavelength_v / dy_m,
            horizontal_wavenumber=kh,
            vertical_wavenumber=m,
            angular_frequency=omega,
        )

    def _check_wave_resolution(self, wave):
        if not self._anomaly_is_active():
            return
        enough_horizontal = (
                wave.points_per_shortest_horizontal_wave
                >= self.MIN_HORIZONTAL_WAVE_POINTS
        )
        enough_vertical = (
                wave.points_per_vertical_wave >= self.MIN_VERTICAL_WAVE_POINTS
        )
        if not (enough_horizontal and enough_vertical):
            raise ValueError(
                "Resolve the anomaly with >=10 horizontal and >=12 vertical "
                "points per shortest wavelength; increase nx or ny"
            )

    def _add_thermal_anomaly(self, state):
        """Change theta and density while retaining P, pressure and velocity."""
        self.applied_thermal_anomaly_max_abs_K = 0.0
        if not self._anomaly_is_active():
            return

        grid = self.config.spatial_grid.grid
        inner = grid.get_inner_slice()
        T_ref = self.config.global_constants.T_ref
        node_shape = self._thermal_wave_nodes(state.nodes_x, state.nodes_y)
        cell_shape = self._nodes_to_cells(node_shape)
        peak = float(np.max(np.abs(cell_shape[inner])))
        if not np.isfinite(peak) or peak <= 0:
            raise ValueError("The grid does not resolve a nonzero thermal wave")

        # Normalizing after averaging makes 0.5 K an actual cell amplitude.
        delta_theta = (self.thermal_anomaly_K / T_ref) * cell_shape / peak
        state.theta = state.theta + delta_theta
        if np.any(state.theta <= 0) or not np.all(np.isfinite(state.theta)):
            raise ValueError("The thermal wave gives invalid theta")

        inverse_theta = 1.0 / state.theta
        state.inverse_theta_anomaly = (
                inverse_theta - state.reference_inverse_theta
        )
        state.rho = state.rho_theta * inverse_theta
        self.applied_thermal_anomaly_max_abs_K = float(
            np.max(np.abs(delta_theta[inner])) * T_ref
        )

    def _check_static_stability(self, theta):
        grid = self.config.spatial_grid.grid
        height_m = np.asarray(grid.y_cells) * self.h_ref
        N2 = self.g_ms2 * np.gradient(
            np.log(theta), height_m, axis=1, edge_order=2
        )
        minimum_N2 = float(np.min(N2[grid.get_inner_slice()]))
        if not np.isfinite(minimum_N2) or minimum_N2 <= 0:
            raise ValueError(
                "Initial stratification overturns; reduce wind or thermal anomaly"
            )
        return minimum_N2

    def _assign_initial_state(self, variables, mpv, state):
        wind = self.config.physics.wind_speed
        fields = variables.cell_vars
        fields[..., VI.RHO] = state.rho
        fields[..., VI.RHOU] = state.rho * (wind[0] + state.u)
        fields[..., VI.RHOV] = 0.0
        fields[..., VI.RHOW] = state.rho * (wind[2] + state.w)
        fields[..., VI.RHOY] = state.rho_theta
        fields[..., VI.RHOX] = state.rho * state.inverse_theta_anomaly
        mpv.p2_nodes[...] = state.pressure_nodes

    def _record_diagnostics(self, state, wave, minimum_N2):
        """Report measured initialization checks and separate linear estimates."""
        grid = self.config.spatial_grid.grid
        inner = grid.get_inner_slice()
        gl = self.config.global_constants
        th = Thermodynamics()
        th.update(gl.gamma)
        Gamma = th.Gamma
        Msq = float(self.config.model_regimes.Msq)
        g = self.config.physics.gravity_strength[1]
        f = self.config.physics.coriolis_strength[1]

        P = state.rho_theta
        rho = state.rho
        gx, gy, gz = state.pressure_gradients
        force_x = (P / Gamma) * gx + f * rho * state.w
        force_y = (P / Gamma) * gy + (g / Msq) * P * state.inverse_theta_anomaly
        force_z = (P / Gamma) * gz - f * rho * state.u
        self.initial_balance_residuals = {
            axis: float(np.max(np.abs(force[inner])))
            for axis, force in zip(("x", "y", "z"), (force_x, force_y, force_z))
        }

        peak_wind = float(
            np.max(np.hypot(state.u[inner], state.w[inner])) * self.velocity_ref
        )
        rossby = peak_wind / (abs(self.f_s) * self.radius_m)
        burger = (
                         self.N_s * self.half_depth_m / (abs(self.f_s) * self.radius_m)
                 ) ** 2

        # Local Boussinesq estimates at the vortex centre, not solver output.
        kh = wave.horizontal_wavenumber
        m = wave.vertical_wavenumber
        omega = wave.angular_frequency
        theta_mid_K = self.T_ref * np.exp(
            self.N_s ** 2 * self.center_height_m / self.g_ms2
        )
        buoyancy_scale = (
                self.g_ms2 * self.applied_thermal_anomaly_max_abs_K / theta_mid_K
        )
        v_scale = buoyancy_scale * kh ** 2 / ((kh ** 2 + m ** 2) * omega)
        theta_departure = state.theta - 1.0 / state.reference_inverse_theta

        self.initial_diagnostics = {
            "mode": self.mode,
            "f_s-1": f / self.t_ref,
            "f_code": f,
            "g_m_s-2": g / Msq * self.h_ref / self.t_ref ** 2,
            "background_N_s-1": self.N_s,
            "peak_wind_m_s-1": peak_wind,
            "Rossby_number": rossby,
            "Burger_number": burger,
            "minimum_N2_s-2": minimum_N2,
            "physical_Exner_amplitude": self.p_amp,
            "horizontal_spacing_m": float(grid.dxyz[0] * self.h_ref),
            "vertical_spacing_m": float(grid.dxyz[1] * self.h_ref),
            "force_residuals": self.initial_balance_residuals,
            "unbalanced_anomaly_enabled": bool(self._anomaly_is_active()),
            "anomaly_kind": self.anomaly_kind,
            "requested_thermal_anomaly_K": self.thermal_anomaly_K,
            "applied_thermal_anomaly_max_abs_K": self.applied_thermal_anomaly_max_abs_K,
            "max_theta_anomaly_K": float(
                np.max(np.abs(theta_departure[inner])) * gl.T_ref
            ),
            "wave_horizontal_wavelength_m": wave.horizontal_wavelength_m,
            "wave_vertical_wavelength_m": wave.vertical_wavelength_m,
            "points_per_shortest_horizontal_wave": float(
                wave.points_per_shortest_horizontal_wave
            ),
            "points_per_vertical_wave": float(wave.points_per_vertical_wave),
            "estimated_wave_period_minutes": float(wave.period_seconds / 60.0),
            "estimated_first_v_peak_minutes": float(wave.period_seconds / 240.0),
            "estimated_linear_v_scale_m_s-1": float(v_scale),
            "requested_duration_minutes": self.duration_hours * 60.0,
            "requested_dt_seconds": self.dt_seconds,
            # Raw force before the diagnostic pressure responds: this is
            # not a constant acceleration to integrate ballistically.
            "max_raw_vertical_force_acceleration_m_s-2": float(
                np.max(np.abs((force_y / rho)[inner]))
                * self.h_ref / self.t_ref ** 2
            ),
        }
        logging.info(
            "Compact vortex / gravity-wave initialization: %s",
            self.initial_diagnostics,
        )
        print("Compact vortex / gravity-wave initialization:", self.initial_diagnostics)

    def dimensional_fields(self, variables: "Variables"):
        """Fields for plotting; slice out grid halos before plotting extrema.

        Plot v_m_s in an x-y slice at the middle z index. A starting colour
        range of +/-0.2 m/s is useful for the default disturbance. Compare
        an on/off pair in the SAME mode to isolate the imposed wave response.
        """
        q = variables.cell_vars
        rho = q[..., VI.RHO]
        u = self.velocity_ref * q[..., VI.RHOU] / rho
        v = self.velocity_ref * q[..., VI.RHOV] / rho
        w = self.velocity_ref * q[..., VI.RHOW] / rho
        dx, _, dz = np.asarray(self.config.spatial_grid.grid.dxyz) * self.h_ref
        return {
            "u_m_s": u,
            "v_m_s": v,
            "w_m_s": w,
            "theta_K": self.config.global_constants.T_ref * q[..., VI.RHOY] / rho,
            "horizontal_divergence_s-1": (
                    np.gradient(u, dx, axis=0, edge_order=2)
                    + np.gradient(w, dz, axis=2, edge_order=2)),
        }