from dataclasses import dataclass, field
import numpy as np
from typing import List, Tuple, Dict, Any
from atmpy.infrastructure.enums import BoundarySide, BoundaryConditions as BdryType
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
    global_constants: GlobalConstants = GlobalConstants()
    spatial_grid: SpatialGrid = SpatialGrid()
    boundary_conditions: BoundaryConditions = BoundaryConditions()
    temporal: Temporal = Temporal()
    model_regimes: ModelRegimes = ModelRegimes()
    physics: Physics = Physics()
    numerics: Numerics = Numerics()
    diagnostics: Diagnostics = Diagnostics()
    outputs: Outputs = Outputs()

    def update_boundary_condition(
        self, boundary_side: BoundarySide, condition: BdryType
    ):
        self.boundary_conditions.conditions[boundary_side] = condition

    def update_from_kwargs(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(getattr(self, key), key, value)
