from typing import Dict, Any, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from atmpy.boundary_conditions.boundary_conditions import (
        BaseBoundaryCondition as BaseBC,
    )
    import numpy as np
from atmpy.infrastructure.factory import get_boundary_conditions
from atmpy.infrastructure.enums import (
    BoundaryConditions as BdryType,
    BoundarySide as BdrySide,
)

from atmpy.infrastructure.utility import side_direction_mapping


class BoundaryManager:
    def __init__(self):
        self.boundary_conditions: Dict[BdrySide, "BaseBC"] = {}

    def setup_conditions(self, bc_dict: Dict[BdrySide, Dict[str, Any]]):
        for side, bc_data in bc_dict.items():
            params = bc_data["params"]
            condition_type = bc_data["type"]
            self._validate_side_direction_compatibility(side=side, **params)
            # Here we use factory function to get the condition object
            bc_instance = get_boundary_conditions(condition_type, side=side, **params)
            self.boundary_conditions[side] = bc_instance
        self._validate_periodic_boundary_condition(side=side, **params)

    def _validate_side_direction_compatibility(
        self, side: BdrySide, **params: Dict[str, Any]
    ):
        """Validates whether the given side is compatible with the given direction"""

        # Check whether the side is compatible with the direction
        direction: int = params["direction"]
        if not side in side_direction_mapping(direction):
            raise ValueError(
                f"{side} is not a valid side for the direction {direction}"
            )

    def _validate_periodic_boundary_condition(self, **kwargs: Dict[str, Any]):
        """
        Validates that in presence of Periodic BC in one side, the opposite side also has periodic BC.

        Raises:
            ValueError: If a periodic condition is not properly paired with its opposite.
        """
        # Define opposites for a cube grid.

        # -----------------------------------------------------------------------------------
        # Validate BOTH sides of the periodic boundary condition have actually periodic boundary conditions

        # Validation only works for periodic boundary condition for now.
        for side, condition in self.boundary_conditions.items():
            if condition.type == BdryType.PERIODIC:
                opposite_side = side.opposite
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

    def apply_single_boundary_condition(self, cells: "np.ndarray", direction: str):
        """Apply the boundary conditions on a single direction. If the boundary condition of the first side is
        PERIODIC, then skip the next side since the condition is automatically applied on the next side too.

        Parameters
        ----------
        cells: np.ndarray
            The variable container.
        direction: str
            The direction to apply the boundary condition on. Values should be "x", "y" or "z".
        """

        sides: Tuple[BdrySide, BdrySide] = side_direction_mapping(direction)
        print(f"Apply boundary conditions on sides: {sides}")
        for side in sides:
            if side in self.boundary_conditions.keys():
                condition = self.boundary_conditions[side]
                condition.apply(cells)
                # Since the periodic boundary condition is automatically applied on both sides, skip the other side
                if condition.type == BdryType.PERIODIC:
                    break

    def apply_full_boundary_conditions(
        self,
        cells,
    ):
        """Apply the boundary conditions on all sides."""
        print("Applying full boundary conditions...")
        for side, condition in self.boundary_conditions.items():
            condition.apply(cells)

    def apply_single_rhs(self, rhs: "np.ndarray", direction: str):
        """Apply the correction on the boundary in a single direction on the source term.

        Parameters
        ----------
        rhs: np.ndarray
            The source variable. The right-hand side of the euler equation.
        direction: str
            The direction of the correction.

        """
        sides = side_direction_mapping(direction)
        print(f"Apply boundary conditions on sides: {sides}")
        for side in sides:
            # if side in self.boundary_conditions.keys():
            condition = self.boundary_conditions[side]
            condition.apply_rhs(rhs)
            # Since the periodic boundary condition is automatically applied on both sides, skip the other side
            if condition.type == BdryType.PERIODIC:
                break

    def apply_all_rhs(self, rhs):
        """Apply boundary corrections to an external source on all sides and directions.

        Parameters
        ----------
        rhs: np.ndarray
            right-hand side of the euler equations. Source terms.
        """
        for side, condition in self.boundary_conditions.items():
            # Only call apply_rhs if the method is implemented.
            if hasattr(condition, "apply_rhs"):
                condition.apply_rhs(rhs)


import numpy as np


def boundary_manager_2d():
    from atmpy.physics.thermodynamics import Thermodynamics
    from atmpy.grid.utility import DimensionSpec, create_grid
    from atmpy.variables.variables import Variables
    from atmpy.infrastructure.enums import VariableIndices as VI
    from atmpy.boundary_conditions.utility import create_params

    np.set_printoptions(linewidth=100)

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
    variables.cell_vars[..., VI.RHO][1:-1, 1:-1] = 4
    variables.cell_vars[..., VI.RHOU] = array
    variables.cell_vars[..., VI.RHOY] = 2
    variables.cell_vars[..., VI.RHOW] = 3

    rng.shuffle(arr)
    array = arr.reshape(nnx, nny)
    variables.cell_vars[..., VI.RHOV] = array
    gravity = np.array([0.0, 1.0, 0.0])
    th = Thermodynamics()

    stratification = lambda x: x**2

    gravity = [0.0, 1.0, 0.0]
    direction = "y"
    bc_dict = {}
    create_params(
        bc_dict,
        BdrySide.TOP,
        BdryType.WALL,
        direction=direction,
        grid=grid,
        gravity=gravity,
        stratification=stratification,
    )
    create_params(
        bc_dict,
        BdrySide.BOTTOM,
        BdryType.WALL,
        direction=direction,
        grid=grid,
        gravity=gravity,
        stratification=stratification,
    )

    manager = BoundaryManager()
    manager.setup_conditions(bc_dict)
    print(manager.boundary_conditions.keys())

    print("Applying boundary conditions for 2D test:")
    print(variables.cell_vars[..., VI.RHOU])
    x = variables.cell_vars[..., VI.RHOV]
    manager.apply_single_boundary_condition(variables.cell_vars, direction)

    # # manager.apply_full_boundary_conditions(variables.cell_vars)
    print(variables.cell_vars[..., VI.RHOV])
    # rhs = np.arange(grid.nnx_total * grid.nny_total).reshape(grid.nshape)
    # print(x)
    # print(np.pad(x[tuple(directional_inner_slice)], pad_width, negative_symmetric))


if __name__ == "__main__":
    boundary_manager_2d()
