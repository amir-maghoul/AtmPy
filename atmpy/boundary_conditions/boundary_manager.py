from typing import Dict, Any
from atmpy.infrastructure.factory import get_boundary_conditions
from atmpy.infrastructure.enums import (
    BoundaryConditions as BdryType,
    BoundarySide as BdrySide,
)
from atmpy.boundary_conditions.boundary_conditions import (
    BaseBoundaryCondition as BaseBC,
)


class BoundaryManager:
    def __init__(self):
        self.boundary_conditions: Dict[BdrySide, BaseBC] = {}

    def setup_conditions(self, bc_dict: Dict[BdrySide, Dict[str, Any]]):
        for side, bc_data in bc_dict.items():
            params = bc_data["params"]
            condition_type = bc_data["type"]
            # Here we use factory function to get the condition object
            bc_instance = get_boundary_conditions(condition_type, side=side, **params)
            self.boundary_conditions[side] = bc_instance
        self.validate_boundary_condition(side=side, **params)

    def validate_boundary_condition(self, **kwargs: Dict[str, Any]):
        """
        Validates the configured boundary conditions.

        In this example, the criterion is that if one side is set to be PERIODIC,
        the opposite side must also be PERIODIC. This prevents mixing a periodic condition
        with a non-periodic one on the other side, which would defeat the purpose of a true periodic domain.

        Raises:
            ValueError: If a periodic condition is not properly paired with its opposite.
        """
        # Define opposites for a cube grid.

        mapping = {
            "x": (BdrySide.LEFT, BdrySide.RIGHT),
            "y": (BdrySide.BOTTOM, BdrySide.TOP),
            "z": (BdrySide.FRONT, BdrySide.BACK),
        }
        # Check whether the side is compatible with the direction
        direction = kwargs["direction"]
        side = kwargs["side"]
        if not side in mapping.get(direction):
            raise ValueError(
                f"{side} is not a valid side for the direction {direction}"
            )

        # -----------------------------------------------------------------------------------
        # Validate BOTH sides of the periodic boundary condition have actually periodic boundary condtions

        opposites = {
            "LEFT": "RIGHT",
            "RIGHT": "LEFT",
            "TOP": "BOTTOM",
            "BOTTOM": "TOP",
            "FRONT": "BACK",
            "BACK": "FRONT",
        }
        # Validation only works for periodic boundary condition for now.
        for side, condition in self.boundary_conditions.items():
            if condition.type == BdryType.PERIODIC:
                opposite_side = opposites.get(side)
                if opposite_side is None:
                    raise ValueError(f"Unknown opposite side for boundary '{side}'.")
                if opposite_side not in self.boundary_conditions:
                    raise ValueError(
                        f"Boundary condition on '{side}' is PERIODIC, but its opposite side "
                        f"'{opposite_side}' is not defined."
                    )
                opposite_condition = self.boundary_conditions[opposite_side]
                if opposite_condition.type != BdryType.PERIODIC:
                    raise ValueError(
                        f"Boundary condition on '{side}' is PERIODIC, but "
                        f"the opposite boundary '{opposite_side}' is set as {opposite_condition.type}."
                    )

    def apply_boundary_conditions(
        self,
        cells,
    ):
        for side, condition in self.boundary_conditions.items():
            condition.apply(cells)


import numpy as np


def boundary_manager_2d():
    from atmpy.physics.thermodynamics import Thermodynamics
    from atmpy.grid.utility import DimensionSpec, create_grid
    from atmpy.variables.variables import Variables
    from atmpy.infrastructure.enums import VariableIndices as VI

    nx = 1
    ngx = 2
    nnx = nx + 2 * ngx
    ny = 2
    ngy = 2
    nny = ny + 2 * ngy

    dim = [DimensionSpec(nx, 0, 2, ngx), DimensionSpec(ny, 0, 2, ngy)]
    grid = create_grid(dim)
    rng = np.random.default_rng()
    arr = np.arange(nnx * nny)
    rng.shuffle(arr)
    array = arr.reshape(nnx, nny)

    variables = Variables(grid, 6, 1)
    variables.cell_vars[..., VI.RHO] = 1
    variables.cell_vars[..., VI.RHOU] = array
    variables.cell_vars[..., VI.RHOY] = 2
    variables.cell_vars[..., VI.RHOW] = 3

    rng.shuffle(arr)
    array = arr.reshape(nnx, nny)
    variables.cell_vars[..., VI.RHOV] = array
    gravity = np.array([0.0, 1.0, 0.0])
    th = Thermodynamics()

    params = {
        "direction": "y",
        "grid": grid,
        "gravity": gravity,
        "stratification": lambda x: x**2,
        "thermodynamics": th,
        "is_lamb": False,
        "is_compressible": True,
    }
    bc_dict = {
        BdrySide.TOP: {"type": BdryType.REFLECTIVE_GRAVITY, "params": params},
    }

    manager = BoundaryManager()
    manager.setup_conditions(bc_dict)
    print(manager.boundary_conditions.keys())

    print("Applying boundary conditions for 2D test:")

    manager.apply_boundary_conditions(variables.cell_vars)
    print(variables.cell_vars[..., VI.RHO])


if __name__ == "__main__":
    boundary_manager_2d()
