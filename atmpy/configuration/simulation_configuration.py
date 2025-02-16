from dataclasses import dataclass, field
import numpy as np
from typing import List, Tuple, Dict, Any
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
)


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

    # # Usable Grid from spatial data
    grid: Grid = field(init=False)

    def __post_init__(self):
        self.grid = self.spatial_grid.grid

    def update_boundary_condition(
        self, boundary_side: BoundarySide, condition: BdryType
    ):
        self.boundary_conditions.conditions[boundary_side] = condition

    def update_from_kwargs(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                for nested_key, nested_value in value.items():
                    setattr(getattr(self, key), nested_key, nested_value)

if __name__ == "__main__":
    x = SimulationConfig()
    print(x.temporal)
