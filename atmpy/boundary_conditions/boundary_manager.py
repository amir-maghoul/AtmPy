from typing import Dict, Any, Tuple, TYPE_CHECKING

from atmpy import grid

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

    def _validate_side_direction_compatibility(
        self, side: BdrySide, **params: Dict[str, Any]
    ):
        """Validates whether the given side is compatible with the given direction"""

        # Check whether the side is compatible with the direction
        direction: str = params["direction"]
        if not side in side_direction_mapping(direction):
            raise ValueError(
                f"{side} is not a valid side for the direction {direction}"
            )

    def apply_boundary_on_one_side(self, cells: "np.ndarray", side: BdrySide):
        """Apply the boundary conditions on a single side.

        Parameters
        ----------
        cells: np.ndarray
            The variable container.
        side: BdrySide
            The side to apply the boundary condition on.
        """

        print(f"Apply boundary conditions *cell variables* on side: {side}")
        if side not in self.boundary_conditions.keys():
            raise ValueError(
                f"The side {side} does not exist in the list of given sides: {self.boundary_conditions.keys()}.")
        condition = self.boundary_conditions[side]
        condition.apply(cells)

    def apply_boundary_on_direction(self, cells: "np.ndarray", direction: str):
        """Apply the boundary conditions on a single direction, consisting of two sides.

        Parameters
        ----------
        cells: np.ndarray
            The variable container.
        direction: str
            The direction to apply the boundary condition on. Values should be "x", "y" or "z".
        """

        sides: Tuple[BdrySide, BdrySide] = side_direction_mapping(direction)
        print(f"Apply boundary conditions *cell variables* on sides: {sides}")
        for side in sides:
            if side in self.boundary_conditions.keys():
                condition = self.boundary_conditions[side]
                condition.apply(cells)

    def apply_boundary_on_all_sides(
        self,
        cells: "np.ndarray"
    ):
        """Apply the boundary conditions on all sides."""
        print("Applying full boundary conditions on *cell variables*...")
        for side, condition in self.boundary_conditions.items():
            condition.apply(cells)

    def apply_boundary_on_single_var_one_side(self, variable: "np.ndarray", side: "BdrySide", is_nodal: bool):
        """Apply the boundary for a single variable on one side.

        Parameters
        ----------
        variable: np.ndarray
            The variable array
        is_nodal: bool
            Whether the variable is nodal or not.
        direction: str
            The direction of the correction.

        """
        print(f"Apply boundary conditions on *cell-valued pressure* on the side: {side}")
        if side not in self.boundary_conditions.keys():
            raise ValueError(f"The side {side} does not exist in the list of given sides: {self.boundary_conditions.keys()}.")
        condition = self.boundary_conditions[side]
        condition.apply_single_variable(variable, is_nodal)

    def apply_boundary_on_single_var_direction(self, variable: "np.ndarray", is_nodal: bool, direction: str):
        """Apply the boundary for a single variable on a single direction consisting of two sides.

        Parameters
        ----------
        variable: np.ndarray
            The variable array
        is_nodal: bool
            Whether the variable is nodal or not.
        direction: str
            The direction of the correction.

        """
        sides = side_direction_mapping(direction)
        print(f"Apply boundary conditions on *cell-valued pressure* on the sides: {sides}")
        for side, condition in self.boundary_conditions.items():
            condition.apply_single_variable(variable, is_nodal)

    def apply_boundary_on_single_var_all_sides(self, variable: "np.ndarray", is_nodal: bool):
        """Apply the boundary conditions on all sides and directions for the input variable.

        Parameters
        ----------
        variable: np.ndarray
            The variable array
        is_nodal: bool
            Whether the variable is nodal or not.
        """
        print("Apply full boundary conditions on *cell-valued pressure*...")
        for side, condition in self.boundary_conditions.items():
            condition.apply_single_variable(variable, is_nodal)


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
        BdrySide.BOTTOM,
        BdryType.WALL,
        direction=direction,
        grid=grid,
        gravity=gravity,
        stratification=stratification,
    )
    # create_params(
    #     bc_dict,
    #     BdrySide.RIGHT,
    #     BdryType.PERIODIC,
    #     direction=direction,
    #     grid=grid,
    #     gravity=gravity,
    #     stratification=stratification,
    # )

    manager = BoundaryManager()
    manager.setup_conditions(bc_dict)
    print(manager.boundary_conditions.keys())

    print("Applying boundary conditions for 2D test:")
    print(variables.cell_vars[..., VI.RHOU])
    x = variables.cell_vars[..., VI.RHOV]
    # manager.apply_boundary_on_direction(variables.cell_vars, direction)
    # print(variables.cell_vars[..., VI.RHOU])


    # # manager.apply_full_boundary_conditions(variables.cell_vars)
    # print(variables.cell_vars[..., VI.RHOU])
    # rhs = np.arange(grid.nnx_total * grid.nny_total).reshape(grid.nshape)
    # print(x)
    # print(np.pad(x[tuple(directional_inner_slice)], pad_width, negative_symmetric))
    manager.apply_boundary_on_single_var_direction(variables.cell_vars[..., VI.RHOU], is_nodal=0, direction=direction)
    print(variables.cell_vars[..., VI.RHOU])


if __name__ == "__main__":
    boundary_manager_2d()
