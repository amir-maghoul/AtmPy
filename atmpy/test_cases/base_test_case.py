"""Module for abstract base class for test cases."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, TYPE_CHECKING
from atmpy.configuration.simulation_configuration import SimulationConfig
from atmpy.infrastructure.enums import BoundarySide, BoundaryConditions as BdryType

if TYPE_CHECKING:
    from atmpy.variables.variables import Variables
    from atmpy.variables.multiple_pressure_variables import MPV


class BaseTestCase(ABC):
    def __init__(self, name: str, config: SimulationConfig):
        self.name = name
        self.config = config
        # Remove parameters, initial_conditions, boundary_conditions attributes
        # Configuration is handled by self.config
        # Initial conditions are handled by initialize_solution
        # Boundary conditions setup is handled during config init or setup

    def set_boundary_condition(
        self,
        boundary_side: BoundarySide,
        condition: BdryType,
    ):
        """Update the boundary condition for a given boundary side in the config."""
        # Modify the config directly
        if boundary_side not in self.config.boundary_conditions.conditions:
            print(
                f"Warning: Boundary side {boundary_side} not initially in config. Adding."
            )
        self.config.boundary_conditions.conditions[boundary_side] = condition

    def set_boundary_conditions(self, bc_updates: Dict[BoundarySide, BdryType]):
        """Update multiple boundary conditions in the configuration."""
        for side, condition in bc_updates.items():
            self.set_boundary_condition(side, condition)

    def set_grid_configuration(self, grid_updates: Dict[str, Any]):
        """Update spatial grid configuration."""
        # This is slightly more complex because the grid object needs recreation
        grid_config = self.config.spatial_grid
        changed = False
        for key, value in grid_updates.items():
            if hasattr(grid_config, key) and getattr(grid_config, key) != value:
                setattr(grid_config, key, value)
                changed = True
        if changed:
            # Re-initialize the grid object within the config
            self.config.spatial_grid.__post_init__()
            # Update the grid reference in the main config too
            self.config.grid = self.config.spatial_grid.grid
            print("Grid configuration updated and grid object recreated.")

    def set_global_constants(self, global_constant_updates: Dict[str, float]):
        """Update the global constants in the configuration."""
        constants_config = self.config.global_constants
        changed = False
        for key, value in global_constant_updates.items():
            if hasattr(constants_config, key):
                setattr(constants_config, key, value)
                changed = True
            else:
                print(f"Warning: Global constant '{key}' not found in config.")
        if changed:
            # Re-run post_init if dependent values need recalculation
            if hasattr(constants_config, "__post_init__"):
                constants_config.__post_init__()

    def set_temporal(self, temporal_updates: Dict[str, Any]):
        """Update the temporal configuration."""
        temporal_config = self.config.temporal
        for key, value in temporal_updates.items():
            if hasattr(temporal_config, key):
                setattr(temporal_config, key, value)
            else:
                print(f"Warning: Temporal setting '{key}' not found in config.")

    def set_physics(self, physics_updates: dict):
        """Update the physics configuration."""
        physics_config = self.config.physics
        for key, value in physics_updates.items():
            if hasattr(physics_config, key):
                setattr(physics_config, key, value)
            else:
                print(f"Warning: Physics setting '{key}' not found in config.")

    def set_model_regimes(self, model_regime_updates: dict):
        """Update the model regime configuration."""
        regime_config = self.config.model_regimes
        for key, value in model_regime_updates.items():
            if hasattr(regime_config, key):
                setattr(regime_config, key, value)
            else:
                print(f"Warning: Model regime '{key}' not found in config.")
        # Recalculate Msq if necessary after updating refs
        self._update_Msq()

    def set_numerics(self, numerics_updates: dict):
        """Update the numerics configuration."""
        numerics_config = self.config.numerics
        for key, value in numerics_updates.items():
            if hasattr(numerics_config, key):
                setattr(numerics_config, key, value)
            else:
                print(f"Warning: Numerics setting '{key}' not found in config.")

    def set_diagnostics(self, diagnostics_updates: dict):
        """Update the diagnostics configuration."""
        diag_config = self.config.diagnostics
        for key, value in diagnostics_updates.items():
            if hasattr(diag_config, key):
                setattr(diag_config, key, value)
            else:
                print(f"Warning: Diagnostics setting '{key}' not found in config.")

    def set_outputs(self, output_updates: dict):
        """Update the output configuration."""
        output_config = self.config.outputs
        for key, value in output_updates.items():
            if hasattr(output_config, key):
                setattr(output_config, key, value)
            else:
                print(f"Warning: Output setting '{key}' not found in config.")
        # Update suffix based on grid if needed
        self._update_output_suffix()

    def _update_Msq(self):
        """Helper to recalculate Msq based on current reference values."""
        const = self.config.global_constants
        regime = self.config.model_regimes
        if const.R_gas > 0 and const.T_ref > 0:
            denominator = const.R_gas * const.T_ref
            regime.Msq = (
                (const.u_ref * const.u_ref) / denominator * regime.is_compressible
            )
        else:
            regime.Msq = 0.0  # Avoid division by zero
            print("Warning: Cannot calculate Msq due to invalid R_gas or T_ref.")

    def _update_output_suffix(self):
        """Helper to update output suffix based on grid size."""
        grid = self.config.spatial_grid
        if grid.ndim == 1:
            self.config.outputs.output_suffix = f"_{grid.nx}"
        elif grid.ndim == 2:
            self.config.outputs.output_suffix = f"_{grid.nx}_{grid.ny}"
        elif grid.ndim == 3:
            self.config.outputs.output_suffix = f"_{grid.nx}_{grid.ny}_{grid.nz}"

    @abstractmethod
    def setup(self):
        """Initialize the test case configuration (grid, boundaries, parameters)."""
        pass

    @abstractmethod
    def initialize_solution(self, variables: "Variables", mpv: "MPV"):
        """Set the initial values for the solution variables (rho, rhoU, rhoY, p2, etc.).

        Parameters
        ----------
        variables : Variables
            The main variable container to be initialized.
        mpv : MPV
            The multiple pressure variable container (for p2 fields).
        """
        pass
