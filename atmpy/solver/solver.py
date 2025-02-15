import numpy as np
from atmpy.solver.utility import boundary_map
from atmpy.infrastructure.factory import get_boundary_conditions
from atmpy.configuration.simulation_configuration import SimulationConfig


class Solver:
    def __init__(
        self,
        config: SimulationConfig,
    ):
        self.ndim = grid.ndim
        self.boundary_conditions = config.boundary_conditions
        self.boundary_map = boundary_map(self.ndim)

    def apply_boundary_conditions(self):
        for bc in self.boundary_conditions:
            if bc.side in self.boundary_map:
                boundary_instance = get_boundary_conditions(bc.type, **bc.params)
                self.boundary_map[bc.side] = boundary_instance
            else:
                raise ValueError(f"Invalid boundary side: {bc.side}")

        for side, bc_instance in self.boundary_map.items():
            if bc_instance:
                self._apply_boundary(side, bc_instance)
            else:
                self._apply_default_boundary(side)

    def _apply_boundary(self, side, bc_instance):
        pass

    def advance_time_step(self):
        pass

    def compute_timestep(self):
        pass

    def run(self):
        pass
