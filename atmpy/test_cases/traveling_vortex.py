from atmpy.test_cases.base_test_case import BaseTestCase

#########################################
# THIS IS JUST A TEMPLATE EXAMPLE FOR NOW
#########################################

# test_cases/traveling_vortex.py
from .base_test_case import BaseTestCase
from boundary_conditions.enums import BoundaryConditions
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class BoundaryFace:
    face_id: int
    normal_vector: Tuple[float, float, float]


class TravelingVortexTestCase(BaseTestCase):
    def __init__(self):
        super().__init__(name="TravelingVortexTestCase")
        self.setup()

    def setup(self):
        self.parameters = {
            'reynolds_number': 1000,
            'vortex_strength': 5.0,
            # Add other relevant parameters
        }
        self.initial_conditions = {
            'velocity_field': ...,  # Define initial velocity field
            'pressure_field': ...,  # Define initial pressure field
            # Add other initial conditions
        }
        self.boundary_conditions = {
            'left': {
                'type': BoundaryConditions.INFLOW,
                'params': {'velocity': 5.0},
                'faces': [BoundaryFace(face_id=1, normal_vector=(-1, 0, 0))]
            },
            'right': {
                'type': BoundaryConditions.OUTFLOW,
                'params': {},
                'faces': [BoundaryFace(face_id=2, normal_vector=(1, 0, 0))]
            },
            'top': {
                'type': BoundaryConditions.SLIP_WALL,
                'params': {},
                'faces': [BoundaryFace(face_id=3, normal_vector=(0, 1, 0))]
            },
            'bottom': {
                'type': BoundaryConditions.SLIP_WALL,
                'params': {},
                'faces': [BoundaryFace(face_id=4, normal_vector=(0, -1, 0))]
            },
        }
