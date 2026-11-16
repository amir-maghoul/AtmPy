"""Weak, vertically uniform rotating vortex in the x-z plane.

Initial pressure weights:
    alpha=0, alpha_const=1: geostrophic (default)
    alpha=1, alpha_const=1: gradient wind
    alpha=1, alpha_const=0: centrifugal only

This case intentionally has zero gravity. Geostrophic initial conditions
retain an O(Rossby-number) centrifugal imbalance in the full momentum equations.
The configured background wind is treated as in the original atmpy case.
"""
from fractions import Fraction
from math import comb, ceil
from typing import TYPE_CHECKING
import logging
import numpy as np

import numpy as np
from scipy.special import beta, betainc
from atmpy.pressure_solver.discrete_operations import ClassicalDiscreteOperator

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

if TYPE_CHECKING:
    from atmpy.variables.variables import Variables
    from atmpy.variables.multiple_pressure_variables import MPV


def traveling_vortex_stratification(y):
    """Constant reference potential-temperature profile; gravity is zero."""
    return 1.0


class TravelingVortex3D(BaseTestCase):
    def __init__(
        self,
        config_override: SimulationConfig = None,
        *,
        size: float = 10.0,
        is_compressible: bool = True,
        alpha: float = 1.0, # Centrifugal force
        alpha_const: float = 1.0, # Geostrophic balance force
    ):
        effective_config = (
            config_override if config_override is not None else SimulationConfig()
        )
        super().__init__(name="TravelingVortex3D", config=effective_config)

        # Vortex parameters; a supplied config remains authoritative for physics
        # and model switches, just as in the original class.

        # --------------------------------------------------------------------------
        # Vortex distribution
        # --------------------------------------------------------------------------
        self.correct_distribution = True

        # --------------------------------------------------------------------------
        # Background wind
        # --------------------------------------------------------------------------
        self.u0 = 1.0
        self.v0 = 0.0
        self.w0 = 1.0

        # --------------------------------------------------------------------------
        # Pressure, density, and balance weights
        # --------------------------------------------------------------------------
        self.p0 = 1.0
        self.rho0 = 0.5
        self.del_rho = 0.5
        self.alpha = float(alpha)
        self.alpha_const = float(alpha_const)
        self.rotdir = 1.0
        self.fac = 1.0

        # --------------------------------------------------------------------------
        # Reference scales and vortex geometry
        # --------------------------------------------------------------------------
        self.h_ref = 10_000.0
        self.t_ref = 100.0
        self.size = float(size)
        if not np.isfinite(self.size) or self.size <= 0:
            raise ValueError("size must be finite and positive")
        self.tmax = 3.0 * size
        self.boxsize = self.R0 = 0.5 * self.size
        self.xc = self.yc = self.zc = 0.0

        # --------------------------------------------------------------------------
        # Physical settings
        # --------------------------------------------------------------------------
        self.baroclinic = self.stratified_atmosphere = False
        self.g = 0.0
        self.coriolis_axis = 1
        self.cor_f = [0.0, 1.0e-4 * self.t_ref, 0.0]


        # --------------------------------------------------------------------------
        # Model regimes and projection
        # --------------------------------------------------------------------------
        self.initial_projection = True
        self.is_nongeostrophic = 1
        self.is_nonhydrostatic = 1
        self.is_compressible = 1
        self.qg_filter = 0.0
        self.qg_filter_number = 2

        # --------------------------------------------------------------------------
        # Time parameters
        # --------------------------------------------------------------------------
        self.dtfixed0 = None
        self.dtfixed = 0.005
        self.acoustic_cfl = True

        # --------------------------------------------------------------------------
        # Output cadence
        # --------------------------------------------------------------------------
        self.checkpoint_freq = 100
        self.output_freq = 1
        self.maxstep = max(10_000, ceil(self.tmax / self.dtfixed) + 1)

        # --------------------------------------------------------------------------
        # Build coefficients and apply configuration
        # --------------------------------------------------------------------------
        self._build_pressure_coefficients()
        if config_override is None:
            self.setup()

    # --------------------------------------------------------------------------
    # Analytical pressure coefficients
    # --------------------------------------------------------------------------
    def _build_pressure_coefficients(self):
        """Integrate the polynomial density times swirl, with exact coefficients.

        rho(s) = rho0 + del_rho*(1-s*s)**6
        F'(s) = rho(s)*s**11*(1-s)**12
        G'(s) = rho(s)*s**6*(1-s)**6
        Thus q = alpha*A**2*[F(s)-F(1)]
               + alpha_const*f*A*R0*[G(s)-G(1)], A=fac*rotdir.
        """
        density = [Fraction(0) for _ in range(13)]
        density[0] = Fraction(str(self.rho0))
        for k in range(7):
            density[2*k] += Fraction(str(self.del_rho)) * (-1)**k * comb(6, k)

        def integral_coefficients(degree, first_power):
            product = [Fraction(0) for _ in range(13 + degree)]
            for j, value in enumerate(density):
                for k in range(degree + 1):
                    product[j+k] += value * (-1)**k * comb(degree, k)
            return np.array(
                [
                    np.longdouble(c.numerator)
                    / np.longdouble(c.denominator)
                    / (first_power + k)
                    for k, c in enumerate(product)
                ],
                dtype=np.longdouble,
            )

        self.coe = integral_coefficients(12, 12)
        self.coe_cor = integral_coefficients(6, 7)
        self.coe_correct = self.coe.copy()
        self.coe_cor_correct = self.coe_cor.copy()

    # --------------------------------------------------------------------------
    # Simulation configuration
    # --------------------------------------------------------------------------
    def setup(self):
        """Configure the SimulationConfig for the 3D Traveling Vortex case."""
        print("Setting up 3D Traveling Vortex configuration...")

        # --------------------------------------------------------------------------
        # Grid Configuration
        # --------------------------------------------------------------------------
        nx = nz = 80
        ny = 3

        grid_updates = {
            "ndim": 3,
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "xmin": -self.boxsize,
            "xmax": self.boxsize,
            "ymin": -self.boxsize,
            "ymax": self.boxsize,
            "zmin": -self.boxsize,
            "zmax": self.boxsize,
            "ngx": 2,
            "ngy": 2,
            "ngz": 2,
        }
        self.set_grid_configuration(grid_updates)

        # --------------------------------------------------------------------------
        # Global Constants
        # --------------------------------------------------------------------------
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

        # --------------------------------------------------------------------------
        # Boundary Conditions
        # --------------------------------------------------------------------------
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

        # --------------------------------------------------------------------------
        # Temporal Setting
        # --------------------------------------------------------------------------
        temporal_updates = {
            "CFL": 0.8,
            "dtfixed": self.dtfixed,
            "dtfixed0": self.dtfixed0,
            "tout": np.array([0.0, self.tmax]),
            "tmax": self.tmax,
            "stepmax": self.maxstep,
            "use_acoustic_cfl": self.acoustic_cfl,
        }
        self.set_temporal(temporal_updates)

        # --------------------------------------------------------------------------
        # Model Regimes
        # --------------------------------------------------------------------------
        regime_updates = {
            "is_nongeostrophic": self.is_nongeostrophic,
            "is_nonhydrostatic": self.is_nonhydrostatic,
            "is_compressible": self.is_compressible,
            "qg_filter": self.qg_filter,
            "qg_filter_number": self.qg_filter_number,
        }
        self.set_model_regimes(regime_updates)

        # --------------------------------------------------------------------------
        # Numerics
        # --------------------------------------------------------------------------
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

        # --------------------------------------------------------------------------
        # Physics Settings
        # --------------------------------------------------------------------------
        cor = self.cor_f
        g = self.g
        physics_updates = {
            "wind_speed": [self.u0, self.v0, self.w0],
            "gravity_strength": (0.0, g, 0.0),  # Gravity is zero for this test case
            "coriolis_strength": cor,
            "stratification": (
                self.constant_stratification
                if not self.baroclinic
                else self.qg_stratification
            ),
        }
        self.set_physics(physics_updates)

        # --------------------------------------------------------------------------
        # Outputs
        # --------------------------------------------------------------------------
        output_updates = {
            "output_type": "test",
            "output_folder": "traveling_vortex_3d",
            "output_base_name": "_traveling_vortex_3d",
            "checkpoint_base_name": "_traveling_vortex_3d_checkpoint",
            "output_timesteps": True,
            "output_frequency_steps": self.output_freq,
            "checkpoint_frequency_steps": self.checkpoint_freq,
        }
        self.set_outputs(output_updates)

        # --------------------------------------------------------------------------
        # Diagnostics
        # --------------------------------------------------------------------------
        diag_updates = {
            "diag": True,
            "diag_current_run": "atmpy_travelling_vortex_3d",
        }
        self.set_diagnostics(diag_updates)

        # --------------------------------------------------------------------------
        #
        # --------------------------------------------------------------------------
        self._update_Msq()
        self._update_output_suffix()

        print(f"Configuration complete. Msq = {self.config.model_regimes.Msq}")
        print(
            f"Output files: {self.config.outputs.output_base_name}{self.config.outputs.output_suffix}"
        )

    # --------------------------------------------------------------------------
    # Reference-state and radial-profile helpers
    # --------------------------------------------------------------------------
    def constant_stratification(self, y):
        return traveling_vortex_stratification(y)

    def qg_stratification(self, y):
        raise NotImplementedError(
            "This initializer is for a vertically uniform, zero-gravity vortex."
        )

    def _reshape_hydrostate_into_3D(
        self, variables, mpv, variable_1d, nodal=False
    ):
        grid = self.config.spatial_grid.grid
        inner = grid.get_inner_slice()
        axis = self.config.physics.gravity.axis
        profile = np.asarray(variable_1d)[inner[axis]]
        shape = [1, 1, 1]
        shape[axis] = profile.size
        target = (
            mpv.p2_nodes[inner].shape
            if nodal
            else variables.cell_vars[inner + (VI.RHO,)].shape
        )
        return np.broadcast_to(profile.reshape(shape), target)

    def _radius(self, x, z, grid):
        dx, dz = x - self.xc, z - self.zc
        Lx, Lz = grid.x_end - grid.x_start, grid.z_end - grid.z_start
        dx = dx - Lx * np.round(dx / Lx)
        dz = dz - Lz * np.round(dz / Lz)
        return dx, dz, np.hypot(dx, dz)

    def _pressure_perturbation(self, radius, coriolis):
        """Physical pressure perturbation q=(p-p0)/Msq, on either grid."""
        result = np.zeros_like(radius, dtype=float)
        mask = radius < self.R0
        s = np.asarray(radius[mask] / self.R0, dtype=np.longdouble)

        def integral_difference(coefficients, first_power):
            # Same coefficient construction on cells and nodes. Extended precision
            # reduces cancellation in the alternating polynomial coefficients.
            value = np.zeros_like(s)
            for k, coefficient in enumerate(coefficients):
                value += coefficient * (s**(first_power + k) - 1)
            return value

        amplitude = np.longdouble(self.fac * self.rotdir)
        q = (
            self.alpha * amplitude**2 * integral_difference(self.coe, 12)
            + self.alpha_const
            * coriolis
            * amplitude
            * self.R0
            * integral_difference(self.coe_cor, 7)
        )
        result[mask] = np.asarray(q, dtype=float)
        return result

    # --------------------------------------------------------------------------
    # Solution initialization
    # --------------------------------------------------------------------------
    def initialize_solution(self, variables: "Variables", mpv: "MPV"):

        # --------------------------------------------------------------------------
        # 1. Grid and physical configuration
        # --------------------------------------------------------------------------
        grid = self.config.spatial_grid.grid
        if grid.ndim != 3:
            raise ValueError("TravelingVortex3D requires a 3D grid")
        gravity = np.asarray(self.config.physics.gravity_strength, dtype=float)
        if np.any(gravity != 0) or self.stratified_atmosphere or self.baroclinic:
            raise NotImplementedError(
                "Nonzero gravity requires a joint hydrostatic/thermodynamic "
                "initialization; this case implements the zero-gravity column."
            )
        wind = np.asarray(self.config.physics.wind_speed, dtype=float)
        if wind.shape != (3,) or wind[1] != 0:
            raise ValueError("Background wind must be horizontal for y-wall boundaries")
        rotation = np.asarray(self.config.physics.coriolis_strength, dtype=float)
        if rotation[0] != 0 or rotation[2] != 0:
            raise ValueError("Coriolis vector must be along the vertical y-axis")
        f = rotation[1]
        if self.R0 <= 0 or self.R0 > 0.5 * min(
            grid.x_end - grid.x_start,
            grid.z_end - grid.z_start,
        ):
            raise ValueError("R0 must be positive and fit inside the periodic box")
        if self.p0 <= 0 or min(self.rho0, self.rho0 + self.del_rho) <= 0:
            raise ValueError("Pressure and density must be positive")

        # --------------------------------------------------------------------------
        # 2. Thermodynamics and hydrostatic reference state
        # --------------------------------------------------------------------------
        th = Thermodynamics()
        th.update(self.config.global_constants.gamma)
        Msq = float(self.config.model_regimes.Msq)
        compressible = bool(self.config.model_regimes.is_compressible)
        if not np.isfinite(Msq) or Msq < 0 or (compressible and Msq == 0):
            raise ValueError("Compressible initialization requires positive finite Msq")
        self._build_pressure_coefficients()
        mpv.state(tuple(gravity), Msq)
        inner = grid.get_inner_slice()

        # --------------------------------------------------------------------------
        # 3. Cell coordinates and vortex velocity
        # --------------------------------------------------------------------------
        # Cell-centered coordinates and velocity, uniform along y.
        XC, YC, ZC = np.meshgrid(
            grid.x_cells[inner[0]],
            grid.y_cells[inner[1]],
            grid.z_cells[inner[2]],
            indexing="ij",
        )
        dx, dz, radius = self._radius(XC, ZC, grid)
        s = radius / self.R0
        inside = radius < self.R0
        rotating = inside & (radius > 0)
        uth = np.zeros_like(radius)
        uth[rotating] = (
            self.rotdir * self.fac * s[rotating]**6 * (1 - s[rotating])**6
        )
        u_pert, w_pert = np.zeros_like(radius), np.zeros_like(radius)
        u_pert[rotating] = uth[rotating] * dz[rotating] / radius[rotating]
        w_pert[rotating] = -uth[rotating] * dx[rotating] / radius[rotating]


        # --------------------------------------------------------------------------
        # 4. Cell density and physical-pressure perturbation
        # --------------------------------------------------------------------------
        rho = np.full_like(radius, self.rho0)
        rho[inside] += self.del_rho * (1 - s[inside]**2)**6
        dp2c = self._pressure_perturbation(radius, f)

        # --------------------------------------------------------------------------
        # 5. Reference thermodynamic fields
        # --------------------------------------------------------------------------
        # Reference P = rho*theta on cells and nodes.
        P0c = self._reshape_hydrostate_into_3D(
            variables, mpv, mpv.hydrostate.cell_vars[..., HI.RHOY0]
        )
        P0n = self._reshape_hydrostate_into_3D(
            variables, mpv, mpv.hydrostate.node_vars[..., HI.RHOY0], nodal=True
        )
        expected_P0 = self.p0**(1 / th.gamma)
        if (
            not np.allclose(P0c, expected_P0, rtol=1e-10, atol=1e-12)
            or not np.allclose(P0n, expected_P0, rtol=1e-10, atol=1e-12)
        ):
            raise ValueError("Hydrostate P0 must match the constant background p0")

        # --------------------------------------------------------------------------
        # 6. Regime-specific cell thermodynamics
        # --------------------------------------------------------------------------
        if compressible:
            p_cells = self.p0 + Msq * dp2c
            if np.any(p_cells <= 0):
                raise ValueError("Vortex produces non-positive cell pressure")
            P = p_cells**(1 / th.gamma)
        else:
            P = P0c

        # --------------------------------------------------------------------------
        # 7. Assign cell variables
        # --------------------------------------------------------------------------
        variables.cell_vars[inner + (VI.RHO,)] = rho
        variables.cell_vars[inner + (VI.RHOU,)] = rho * (wind[0] + u_pert)
        variables.cell_vars[inner + (VI.RHOV,)] = rho * wind[1]
        variables.cell_vars[inner + (VI.RHOW,)] = rho * (wind[2] + w_pert)
        variables.cell_vars[inner + (VI.RHOY,)] = P
        # Zero gravity: this buoyancy variable has no dynamical effect.
        variables.cell_vars[inner + (VI.RHOX,)] = 0.0

        # --------------------------------------------------------------------------
        # 8. Nodal pressure initialization
        # --------------------------------------------------------------------------
        # Nodal pressure: same physical q profile, evaluated at node coordinates.
        XN, YN, ZN = np.meshgrid(
            grid.x_nodes[inner[0]],
            grid.y_nodes[inner[1]],
            grid.z_nodes[inner[2]],
            indexing="ij",
        )
        _, _, radius_nodes = self._radius(XN, ZN, grid)
        p2_nodes_unscaled = self._pressure_perturbation(radius_nodes, f)
        if compressible:
            relative_dp = Msq * p2_nodes_unscaled / self.p0
            if np.any(relative_dp <= -1):
                raise ValueError("Vortex produces non-positive nodal pressure")
            # Exact pi2 = [(p0 + Msq*q)**Gamma - p0**Gamma]/Msq.
            mpv.p2_nodes[inner] = (
                self.p0**th.Gamma
                * np.expm1(th.Gamma * np.log1p(relative_dp))
                / Msq
            )
        else:
            mpv.p2_nodes[inner] = th.Gamma * p2_nodes_unscaled / P0n

        # --------------------------------------------------------------------------
        # 9. Completion
        # --------------------------------------------------------------------------
        # Ghost cells are filled by the normal boundary-manager startup sequence.
        logging.info(
            "3D vortex initialized: compressible=%s, alpha=%g, "
            "alpha_const=%g, fac=%g, R0=%g, f=%g",
            compressible,
            self.alpha,
            self.alpha_const,
            self.fac,
            self.R0,
            f,
        )