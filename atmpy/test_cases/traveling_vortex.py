from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any
from atmpy.test_cases.base_test_case import BaseTestCase
from atmpy.configuration.simulation_configuration import SimulationConfig
from atmpy.configuration.simulation_data import BoundaryFace
from atmpy.infrastructure.enums import BoundaryConditions as BdryType, BoundarySide


@dataclass
class TravelingVortexConfig(SimulationConfig):
    pass


class TravelingVortex(BaseTestCase):
    def __init__(self):
        super().__init__(name="TravelingVortexTestCase", config=TravelingVortexConfig())
        self.setup()

    def setup(self):
        self.set_boundary_condition(
            BoundarySide.LEFT,
            BdryType.INFLOW,
            params={"velocity": 5.0},
            faces=[BoundaryFace(face_id=1, normal_vector=(-1, 0, 0))],
        )
        self.set_boundary_condition(
            BoundarySide.RIGHT,
            BdryType.OUTFLOW,
            params={},
            faces=[BoundaryFace(face_id=2, normal_vector=(1, 0, 0))],
        )
        self.set_boundary_condition(
            BoundarySide.TOP,
            BdryType.SLIP_WALL,
            params={},
            faces=[BoundaryFace(face_id=3, normal_vector=(0, 1, 0))],
        )
        self.set_boundary_condition(
            BoundarySide.BOTTOM,
            BdryType.SLIP_WALL,
            params={},
            faces=[BoundaryFace(face_id=4, normal_vector=(0, -1, 0))],
        )
