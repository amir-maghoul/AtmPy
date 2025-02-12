from typing import Dict, Any
from atmpy.boundary_conditions.boundary_conditions import (
    BoundaryCondition, SlipWall, InflowBoundary, OutflowBoundary
)
# boundary_conditions/boundary_condition_manager.py
from atmpy.data._factory import get_boundary_conditions


class BoundaryConditionManager:
    def __init__(self):
        self.boundary_conditions: Dict[str, Any] = {}

    def setup_conditions(self, boundary_conditions: Dict[str, Dict[str, Any]]):
        for side, bc_data in boundary_conditions.items():
            condition_type = bc_data['type']
            params = bc_data.get('params', {})
            faces = bc_data.get('faces', [])
            # Instantiate boundary condition using factory
            bc_instance = get_boundary_conditions(condition_type, **params, faces=faces)
            self.boundary_conditions[side] = bc_instance

    def apply_boundary_conditions(self, cells, faces, solver_state):
        for side, condition in self.boundary_conditions.items():
            condition.apply(cells, faces, solver_state)

