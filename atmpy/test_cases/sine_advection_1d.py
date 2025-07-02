import logging

import numpy as np
from typing import TYPE_CHECKING

from atmpy.boundary_conditions.contexts import BCApplicationContext
from atmpy.test_cases.base_test_case import BaseTestCase
from atmpy.configuration.simulation_configuration import SimulationConfig
from atmpy.infrastructure.enums import (
    BoundaryConditions as BdryType,
    BoundarySide,
    AdvectionRoutines,
    VariableIndices as VI,
    SlopeLimiters as LimiterType,
    # Add RiemannSolvers if you want to make it configurable
)
from atmpy.physics.thermodynamics import (
    Thermodynamics,
)  # For Msq calculation, though not strictly needed for this simple case

if TYPE_CHECKING:
    from atmpy.variables.variables import Variables
    from atmpy.variables.multiple_pressure_variables import MPV


class SineWaveAdvection1D(BaseTestCase):
    """
    1D Advection of a sine wave.
    - Constant advection speed.
    - Periodic boundary conditions.
    - Only advects density (rho) and a passive scalar (rhoY acting as a tracer).
      Momenta (rhou, rhov, rhow) and rhoX will be kept minimal or zero.
    """

    def __init__(self):
        super().__init__(name="SineWaveAdvection1D", config=SimulationConfig())

        # Test Case Specific Parameters
        self.advection_speed: float = 2.0  # u0
        self.wavelength: float = 1.0  # Wavelength of the sine wave
        self.amplitude: float = 0.5  # Amplitude of the sine wave perturbation
        self.background_rho: float = 1.0  # Background density
        self.background_rhoY: float = 3.0  # Background for the 'tracer' rhoY

        self.setup()

    def setup(self):
        """Configure the SimulationConfig for the 1D Sine Wave Advection case."""
        print("Setting up 1D Sine Wave Advection configuration...")

        # --- Grid Configuration ---
        grid_updates = {
            "ndim": 1,
            "nx": 100,  # Number of cells in x
            "ny": 0,
            "nz": 0,
            "xmin": 0.0,
            "xmax": self.wavelength * 2.0,  # Domain length (e.g., 2 wavelengths)
            "ymin": 0.0,  # Not used but required by DimensionSpec
            "ymax": 1.0,  # Not used
            "ngx": 2,  # Ghost cells for MUSCL
            "ngy": 0,  # No y-dimension
        }
        self.set_grid_configuration(grid_updates)

        # --- Boundary Conditions ---
        self.set_boundary_condition(
            BoundarySide.LEFT, BdryType.PERIODIC, mpv_type=BdryType.PERIODIC
        )
        self.set_boundary_condition(
            BoundarySide.RIGHT, BdryType.PERIODIC, mpv_type=BdryType.PERIODIC
        )
        # No TOP/BOTTOM for 1D

        # --- Temporal Settings ---
        # One period T = Domain Length / Advection Speed
        domain_length = grid_updates["xmax"] - grid_updates["xmin"]
        one_period = domain_length / self.advection_speed
        temporal_updates = {
            "CFL": 0.4,  # Typical CFL for stability with MUSCL
            "dtfixed": 0.005,  # Will be overridden by CFL if use_cfl_dt is true
            "use_cfl_dt": True,
            "tmax": one_period * 1.0,  # Simulate for one full advection period
            "tout": np.linspace(0, one_period * 1.0, 11),  # Output 10 frames + initial
            "stepmax": 10000,
        }
        self.set_temporal(temporal_updates)

        # --- Physics Settings ---
        physics_updates = {
            "wind_speed": [
                self.advection_speed,
                0.0,
                0.0,
            ],  # Background flow is the advection speed
            "gravity_strength": (0.0, 0.0, 0.0),  # No gravity
            "coriolis_strength": (0.0, 0.0, 0.0),  # No Coriolis
            "stratification": lambda y: 1.0,  # Not relevant for 1D advection of rho,rhoY
        }
        self.set_physics(physics_updates)

        # --- Model Regimes ---
        # For simple advection, we can effectively turn off pressure effects.
        # is_compressible = 0 would make dp2/dt = 0 if pressure equation is simple.
        # Msq can be set high to minimize sound speed effects if we were doing full Euler.
        # For now, let's keep it simple and focus on the advection part.
        # The IMEX scheme will still run the pressure solver, but its impact should be minimal
        # if the flow is divergence-free (which u=const is).
        regime_updates = {
            "is_nongeostrophic": 1,  # Not relevant
            "is_nonhydrostatic": 1,  # Not relevant
            "is_compressible": 0,  # Try to minimize pressure effects for pure advection
            # Set to 1 if you want to test pressure solver interaction
            "Msq": 1.0e6,  # Make sound speed very high to decouple from advection
        }
        self.set_model_regimes(regime_updates)

        # --- Numerics ---
        numerics_updates = {
            "limiter_scalars": LimiterType.VAN_LEER,  # Or MINMOD for more diffusion
            # Use FIRST_ORDER_RK for the predictor in IMEX, or a simpler advection routine if available
            "first_order_advection_routine": AdvectionRoutines.FIRST_ORDER_RK,
            "second_order_advection_routine": AdvectionRoutines.STRANG_SPLIT,  # For the corrector
            "initial_projection": False,  # Not needed for this simple setup
            # Add Riemann solver choice if desired
        }
        self.set_numerics(numerics_updates)

        # --- Outputs ---
        output_updates = {
            "output_type": "test",  # Keep as 'test' for simple .nc output
            "output_folder": "sine_wave_advection_1d",
            "output_base_name": "_sine_wave_1d",
            "output_timesteps": True,
            "output_frequency_steps": 1,  # Output every step for debugging, or less frequent
            "variables_to_output": ["RHO", "RHOU", "RHOV", "RHOY"],  # Focus on these
        }
        self.set_outputs(output_updates)

        # --- Diagnostics ---
        diag_updates = {
            "diag": False,  # Turn off complex diagnostics for this simple case
        }
        self.set_diagnostics(diag_updates)

        # --- Global Constants ---
        # These are less critical for pure advection but fill them in.
        constants_updates = {
            "gamma": 1.4,
            "R_gas": 287.4,
            "p_ref": 1.0e5,
            "T_ref": 300.0,
            "h_ref": 10000.0,
            "t_ref": 100.0,
            "grav": 0.0,
        }
        self.set_global_constants(constants_updates)

        self._update_Msq()  # Recalculate Msq if regime_updates or constants changed it.
        self._update_output_suffix()

        print(
            f"1D Sine Wave Advection configuration complete. Msq = {self.config.model_regimes.Msq}"
        )
        print(
            f"Output files: {self.config.outputs.output_base_name}{self.config.outputs.output_suffix}"
        )

    def initialize_solution(self, variables: "Variables", mpv: "MPV"):
        """Initialize density (rho) and rhoY with a sine wave."""
        print("Initializing solution for 1D Sine Wave Advection...")

        grid = self.config.grid
        inner_slice_x = grid.get_inner_slice()  # Slice for x-direction inner cells

        # --- Cell-Centered Coordinates (only x matters) ---
        # XC will be a 1D array of cell centers
        XC = grid.x_cells[inner_slice_x]  # Get inner cell centers for x

        # --- Sine Wave Calculation ---
        # k = 2 * pi / wavelength
        wavenumber = 2 * np.pi / self.wavelength
        sine_perturbation = self.amplitude * np.sin(wavenumber * XC)

        # --- Initialize Variables (Inner Domain Only) ---
        # VI.RHO
        rho_initial = self.background_rho + sine_perturbation
        variables.cell_vars[inner_slice_x + (VI.RHO,)] = rho_initial

        # VI.RHOU (rho * u)
        # u is constant (self.advection_speed)
        # We assume the `wind_speed` in physics config is the actual velocity here.
        # The advection routines should use the primitive velocity `u` for reconstruction.
        # The `flux.apply_reconstruction` gets `self.variables`, from which it derives primitives.
        # So, rhou = rho_initial * self.advection_speed
        variables.cell_vars[inner_slice_x + (VI.RHOU,)] = (
            rho_initial * self.advection_speed
        )

        # VI.RHOV, VI.RHOW (should be zero for 1D)
        if grid.ndim > 1 and VI.RHOV < variables.num_vars_cell:  # Check if RHOV exists
            variables.cell_vars[
                inner_slice_x + (slice(None),) * (grid.ndim - 1) + (VI.RHOV,)
            ] = 0.0
        if grid.ndim > 2 and VI.RHOW < variables.num_vars_cell:  # Check if RHOW exists
            variables.cell_vars[
                inner_slice_x + (slice(None),) * (grid.ndim - 1) + (VI.RHOW,)
            ] = 0.0

        # VI.RHOY (acting as a passive tracer, with its own sine wave)
        # Let's make it slightly different from rho for clarity, e.g., different amplitude or phase
        # Or just use the same perturbation on its background
        rhoY_initial = (
            sine_perturbation + self.background_rhoY
        )  # Could be different: 0.8*sine_perturbation
        variables.cell_vars[inner_slice_x + (VI.RHOY,)] = rhoY_initial

        # VI.RHOX (another passive tracer, set to constant or zero)
        if VI.RHOX < variables.num_vars_cell:  # Check if RHOX exists
            variables.cell_vars[inner_slice_x + (VI.RHOX,)] = 0.0  # Or some constant

        # --- Initialize MPV (Pressure variables) ---
        # For this advection test, pressure is ideally constant.
        # Hydrostatic balance with zero gravity should result in constant pressure.
        # Msq is high, so dP/dpi is small, meaning pi changes have little effect on P.
        # If is_compressible = 0, then p2 update from divergence is also zero.
        gravity = self.config.physics.gravity_strength
        Msq = self.config.model_regimes.Msq
        mpv.state(
            gravity, Msq
        )  # This calculates S0, S, hydrostate.cell_vars and hydrostate.node_vars

        # Initialize p2_nodes and p2_cells to zero (no perturbation initially)
        mpv.p2_nodes[...] = 0.0
        mpv.p2_cells[...] = 0.0
        mpv.dp2_nodes[...] = 0.0  # Also initialize the increment

        logging.info("1D Sine Wave Advection solution initialization complete.")


if __name__ == "__main__":
    from atmpy.variables.variables import Variables
    from atmpy.variables.multiple_pressure_variables import MPV
    from atmpy.boundary_conditions.boundary_manager import BoundaryManager
    import matplotlib.pyplot as plt

    # --- Instantiate and Setup ---
    case = SineWaveAdvection1D()
    config = case.config  # Get the configured SimulationConfig

    # --- Create Necessary Simulation Components ---
    grid = config.grid
    mpv = MPV(grid)
    num_vars_cell = (
        max(VI.RHO, VI.RHOU, VI.RHOV, VI.RHOW, VI.RHOX, VI.RHOY) + 1
    )  # Ensure enough vars
    variables = Variables(grid, num_vars_cell=num_vars_cell, num_vars_node=1)

    # --- Initialize Solution using Test Case ---
    case.initialize_solution(variables, mpv)

    # --- Boundary Manager and Apply Initial BCs ---
    bm_config = config.get_boundary_manager_config()
    manager = BoundaryManager(bm_config)
    manager.apply_boundary_on_all_sides(variables.cell_vars)
    # For 1D, only left/right contexts needed for p2_nodes
    nodal_bc_contexts_1d = [BCApplicationContext(is_nodal=True)] * grid.ndim * 2
    # Need to call the specific apply_boundary_on_single_var_direction
    # For simplicity in testing, let's assume apply_boundary_on_single_var_all_sides handles it
    # or we manually apply for x:
    manager.apply_boundary_on_single_var_direction(
        mpv.p2_nodes, "x", nodal_bc_contexts_1d
    )

    # --- Plot Initial Condition ---
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

    inner_slice_x = grid.get_inner_slice()
    x_coords_full = grid.x_cells  # Full grid including ghosts
    x_coords_inner = grid.x_cells[inner_slice_x]  # Inner grid

    # Plot RHO
    axs[0].plot(
        x_coords_full,
        variables.cell_vars[:, VI.RHO],
        "bo-",
        label="Rho (full domain)",
        markersize=3,
    )
    axs[0].plot(
        x_coords_inner,
        variables.cell_vars[tuple(inner_slice_x) + (VI.RHO,)],
        "r.--",
        label="Rho (inner domain)",
    )
    axs[0].set_ylabel("Density (rho)")
    axs[0].set_title("Initial Condition: 1D Sine Wave Advection")
    axs[0].legend()
    axs[0].grid(True)

    # Plot RHOY
    axs[1].plot(
        x_coords_full,
        variables.cell_vars[:, VI.RHOY],
        "go-",
        label="RhoY (full domain)",
        markersize=3,
    )
    axs[1].plot(
        x_coords_inner,
        variables.cell_vars[tuple(inner_slice_x) + (VI.RHOY,)],
        "m.--",
        label="RhoY (inner domain)",
    )
    axs[1].set_ylabel("Tracer (rhoY)")
    axs[1].set_xlabel("X coordinate")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

    # You can now use this 'case' object in your main.py or a simplified solver script.
    # Example:
    # args.case = "SineWaveAdvection1D" # If using command line arguments
    # if args.case == "SineWaveAdvection1D":
    #     case = SineWaveAdvection1D()
