from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any
from atmpy.test_cases.base_test_case import BaseTestCase
from atmpy.configuration.simulation_configuration import SimulationConfig
from atmpy.configuration.simulation_data import BoundaryFace, Temporal
from atmpy.infrastructure.enums import BoundaryConditions as BdryType, BoundarySide


class TravelingVortex(BaseTestCase):
    def __init__(self):
        super().__init__(name="TravelingVortexTestCase", config=SimulationConfig())
        self.setup()

        physics = {
            "u_wind_speed": 1.0,
            "v_wind_speed": 0.0,
            "w_wind_speed": 0.0,
            "stratification": lambda y: 1.0,
        }

        self.set_physics(physics)
        self.config.temporal.CFL = 0.0

    def setup(self):
        self.set_boundary_condition(
            BoundarySide.LEFT,
            BdryType.INFLOW,
        )
        self.set_boundary_condition(
            BoundarySide.RIGHT,
            BdryType.OUTFLOW,
        )
        self.set_boundary_condition(
            BoundarySide.TOP,
            BdryType.SLIP_WALL,
        )
        self.set_boundary_condition(
            BoundarySide.BOTTOM,
            BdryType.SLIP_WALL,
        )


if __name__ == "__main__":
    x = TravelingVortex()
    print(x.boundary_conditions[BoundarySide.LEFT])
    print(x.config.boundary_conditions.conditions)
