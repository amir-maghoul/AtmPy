"""This module handles and updates the simulation configuration of the problem"""

from dataclasses import dataclass, field
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from atmpy.boundary_conditions.contexts import (
    BCInstantiationOptions,
    BoundaryConditionsConfiguration,
)
from atmpy.infrastructure.enums import BoundarySide, BoundaryConditions as BdryType
from atmpy.grid.kgrid import Grid
from atmpy.configuration.simulation_data import (
    GlobalConstants,
    SpatialGrid,
    BoundaryConditions,
    Temporal,
    ModelRegimes,
    Physics,
    Numerics,
    Diagnostics,
    Outputs,
    BoundarySpec,
)
from atmpy.infrastructure.utility import direction_axis
from atmpy.physics.thermodynamics import Thermodynamics


@dataclass
class SimulationConfig:
    global_constants: GlobalConstants = field(default_factory=GlobalConstants)
    spatial_grid: SpatialGrid = field(default_factory=SpatialGrid)
    boundary_conditions: BoundaryConditions = field(default_factory=BoundaryConditions)
    temporal: Temporal = field(default_factory=Temporal)
    model_regimes: ModelRegimes = field(default_factory=ModelRegimes)
    physics: Physics = field(default_factory=Physics)
    numerics: Numerics = field(default_factory=Numerics)
    diagnostics: Diagnostics = field(default_factory=Diagnostics)
    outputs: Outputs = field(default_factory=Outputs)

    def __post_init__(self):
        self.update_all_derived_fields()

    def update_all_derived_fields(self):
        """Central method to update all dependent configurations."""
        self.global_constants.update_global_constants()
        self.physics.update_derived_fields(self.global_constants, self.spatial_grid)
        self.model_regimes.update_derived_fields(self.global_constants)

    def update_boundary_condition(
        self,
        boundary_side: BoundarySide,
        main_type: BdryType,
        mpv_type: Optional[BdryType] = None,
    ):
        """Updates the boundary condition specification for a given side."""
        if boundary_side not in self.boundary_conditions.conditions:
            print(
                f"Warning: Adding new boundary specification for side {boundary_side}"
            )
        self.boundary_conditions.conditions[boundary_side] = BoundarySpec(
            main_type=main_type, mpv_type=mpv_type
        )

    def update_boundary_conditions(self, bc_specs: Dict[BoundarySide, BoundarySpec]):
        """Updates multiple boundary condition specifications."""
        self.boundary_conditions.conditions.update(bc_specs)

    def get_boundary_manager_config(
        self, mpv: "MPV" = None
    ) -> BoundaryConditionsConfiguration:
        """
        Generates the configuration object needed to instantiate the BoundaryManager.

        Parameters
        ----------
        mpv : MPV
            The MPV needed for some BC.
        """
        options_list: List[BCInstantiationOptions] = []
        grid = self.spatial_grid.grid  # Use the grid object from self

        # Determine direction for each side (assuming standard Cartesian mapping)
        side_to_direction_map = {
            BoundarySide.LEFT: "x",
            BoundarySide.RIGHT: "x",
            BoundarySide.BOTTOM: "y",
            BoundarySide.TOP: "y",
            BoundarySide.FRONT: "z",
            BoundarySide.BACK: "z",
        }

        for side, spec in self.boundary_conditions.conditions.items():
            # Skip if side doesn't match grid dimension (e.g., FRONT/BACK in 2D)
            direction = side_to_direction_map.get(side)
            if direction is None:
                continue  # Should not happen with standard enums
            # get the index of the
            direction_index = direction_axis(direction)
            if direction_index >= grid.ndim:
                continue  # This side doesn't exist in this grid dimensionality

            # Create the options object, passing necessary context from self.config
            th: Thermodynamics = Thermodynamics()
            th.update(self.global_constants.gamma)
            self.update_all_derived_fields()
            opts = BCInstantiationOptions(
                side=side,
                type=spec.main_type,
                mpv_boundary_type=spec.mpv_type,  # Pass the MPV type
                direction=direction,
                grid=grid,
                gravity=self.physics.gravity_strength,
                stratification=self.physics.stratification,
                thermodynamics=th,
                is_compressible=bool(self.model_regimes.is_compressible),
                mpv=mpv,
            )
            options_list.append(opts)

        if not options_list:
            raise ValueError(
                "No valid boundary conditions generated for the BoundaryManager configuration."
            )

        return BoundaryConditionsConfiguration(options=options_list)

    def update_from_kwargs(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                for nested_key, nested_value in value.items():
                    setattr(getattr(self, key), nested_key, nested_value)


if __name__ == "__main__":
    x = SimulationConfig()
    print(x.temporal)
