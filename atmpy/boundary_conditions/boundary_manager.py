from typing import Dict, Any, Tuple, TYPE_CHECKING

from mypy.stubtest import get_mypy_type_of_runtime_value

if TYPE_CHECKING:
    from atmpy.boundary_conditions.boundary_conditions import (
        BaseBoundaryCondition as BaseBC,
    )
    import numpy as np
    from atmpy.boundary_conditions.contexts import (
        BoundaryConditionsConfiguration,
        BCApplicationContext,
    )
from atmpy.infrastructure.factory import get_boundary_conditions
from atmpy.infrastructure.enums import (
    BoundaryConditions as BdryType,
    BoundarySide as BdrySide,
)
from atmpy.infrastructure.utility import side_direction_mapping

class BoundaryManager:
    """ The Boundary manager class to apply boundary condition as the strategy."""
    def __init__(self, bc_config: "BoundaryConditionsConfiguration") -> None:
        # The keys will be the boundary sides and the values are the created BC instances.
        self.boundary_conditions: Dict[BdrySide, "BaseBC"] = {}
        self.setup_conditions(bc_config)

    def setup_conditions(self, bc_config: "BoundaryConditionsConfiguration") -> None:
        """ Set up the boundary condition in a dictionary."""
        for inst_opts in bc_config.options:
            # Validate that the given side is compatible with the direction.
            BoundaryManager._validate_side_direction_compatibility(
                inst_opts.side, inst_opts.direction
            )
            # Create the BC instance using the factory.
            bc_instance = get_boundary_conditions(inst_opts.type, inst_opts)
            # fill the boundary_condition dictionary with the needed data
            self.boundary_conditions[inst_opts.side] = bc_instance

    @staticmethod
    def _validate_side_direction_compatibility(
        side: BdrySide, direction: str
    ) -> None:
        """
        Validates whether the given boundary side is compatible with the provided direction.
        """
        if side not in side_direction_mapping(direction):
            raise ValueError(
                f"{side} is not a valid side for the direction {direction}"
            )

    def apply_boundary_on_one_side(
        self,
        cells: "np.ndarray",
        side: BdrySide,
        context: "BCApplicationContext" = None,
    ):
        """Apply the boundary conditions on a single side.

        Parameters
        ----------
        cells: np.ndarray
            The variable container.
        side: BdrySide
            The side to apply the boundary condition on.
        context: BCApplicationContext
            The context object containing the apply method information.
        """

        print(f"Apply boundary conditions *cell variables* on side: {side}")
        if side not in self.boundary_conditions.keys():
            raise ValueError(
                f"The side {side} does not exist in the list of given sides: {self.boundary_conditions.keys()}."
            )
        condition = self.boundary_conditions[side]
        condition.apply(cells, context)

    def apply_boundary_on_direction(
        self,
        cells: "np.ndarray",
        direction: str,
        context: "BCApplicationContext" = None,
    ):
        """Apply the boundary conditions on a single direction, consisting of two sides.

        Parameters
        ----------
        cells: np.ndarray
            The variable container.
        direction: str
            The direction to apply the boundary condition on. Values should be "x", "y" or "z".
        context: BCApplicationContext
            The context object containing the apply method information.
        """

        sides: Tuple[BdrySide, BdrySide] = side_direction_mapping(direction)
        print(f"Apply boundary conditions *cell variables* on sides: {sides}")
        for side in sides:
            if side in self.boundary_conditions.keys():
                condition = self.boundary_conditions[side]
                condition.apply(cells, context)

    def apply_boundary_on_all_sides(
        self, cells: "np.ndarray", context: "BCApplicationContext" = None
    ):
        """Apply the boundary conditions on all sides."""
        print("Applying full boundary conditions on *cell variables*...")
        for side, condition in self.boundary_conditions.items():
            condition.apply(cells, context)

    def apply_boundary_on_single_var_one_side(
        self,
        variable: "np.ndarray",
        side: "BdrySide",
        context: "BCApplicationContext" = None,
    ):
        """Apply the boundary for a single variable on one side.

        Parameters
        ----------
        variable: np.ndarray
            The variable array
        side: BdrySide
            The side to apply the boundary condition on.
        context: BCApplicationContext
            The context object containing the apply method information.
        """

        print(
            f"Apply BC on single variable on the side: {side}"
        )
        if side not in self.boundary_conditions.keys():
            raise ValueError(
                f"The side {side} does not exist in the list of given sides: {self.boundary_conditions.keys()}."
            )
        condition = self.boundary_conditions[side]
        condition.apply_single_variable(variable, context)

    def apply_boundary_on_single_var_direction(
        self,
        variable: "np.ndarray",
        direction: str,
        context: "BCApplicationContext" = None,
    ):
        """Apply the boundary for a single variable on a single direction consisting of two sides.

        Parameters
        ----------
        variable: np.ndarray
            The variable array
        direction: str
            The direction of the correction.
        context: BCApplicationContext
            The context object containing the apply method information.
        """
        sides = side_direction_mapping(direction)
        print(
            f"Apply BC on single variable on the sides: {sides}"
        )
        for side, condition in self.boundary_conditions.items():
            condition.apply_single_variable(variable, context)

    def apply_boundary_on_single_var_all_sides(
        self, variable: "np.ndarray", context: "BCApplicationContext"
    ):
        """Apply the boundary conditions on all sides and directions for the input variable.

        Parameters
        ----------
        variable: np.ndarray
            The variable array
        context: BCApplicationContext
            The context object containing the apply method information.
        """
        print("Apply BC on single variable on all sides...")
        for side, condition in self.boundary_conditions.items():
            condition.apply_single_variable(variable, context)

    def apply_extra(self, variable: "np.ndarray", side: "BdrySide", context: "BCApplicationContext") -> None:
        """ Apply the extra conditions on the given single variable for a single side.

        Parameters
        ----------
        variable: np.ndarray
            The variable array
        side: BdrySide
            The side to apply the boundary condition on.
        context: BCApplicationContext
            The context object containing the apply method information.
        """

        print(
            f"Apply EXTRA boundary conditions on on the side: {side}"
        )
        if side not in self.boundary_conditions.keys():
            raise ValueError(
                f"The side {side} does not exist in the list of given sides: {self.boundary_conditions.keys()}."
            )
        condition = self.boundary_conditions[side]
        condition.apply_extra(variable, context)


import numpy as np


def boundary_manager_2d():
    from atmpy.physics.thermodynamics import Thermodynamics
    from atmpy.grid.utility import DimensionSpec, create_grid
    from atmpy.variables.variables import Variables
    from atmpy.infrastructure.enums import VariableIndices as VI
    from atmpy.boundary_conditions.utility import create_params

    np.set_printoptions(linewidth=100)
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=10)

    nx = 6
    ngx = 2
    nnx = nx + 2 * ngx
    ny = 5
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
    from atmpy.boundary_conditions.contexts import (
        BCInstantiationOptions,
        BoundaryConditionsConfiguration,
        BCApplicationContext,
        ReflectiveGravityBCInstantiationOptions as RFBCInstantiationOptions,
    )

    bc = BCInstantiationOptions(
        side=BdrySide.BOTTOM, type=BdryType.WALL, direction=direction, grid=grid
    )
    bc2 = RFBCInstantiationOptions(side=BdrySide.TOP, type=BdryType.WALL, direction=direction, grid=grid)
    bc3 = RFBCInstantiationOptions(side=BdrySide.LEFT, type=BdryType.WALL, direction="x", grid=grid)
    bc4 = RFBCInstantiationOptions(side=BdrySide.RIGHT, type=BdryType.WALL, direction="x", grid=grid)
    options = [bc, bc2, bc3, bc4]
    bc_conditions = BoundaryConditionsConfiguration(options)
    # create_params(
    #     bc_dict,
    #     BdrySide.RIGHT,
    #     BdryType.PERIODIC,
    #     direction=direction,
    #     grid=grid,
    #     gravity=gravity,
    #     stratification=stratification,
    # )

    manager = BoundaryManager(bc_conditions)
    print(manager.boundary_conditions.keys())

    print("Applying boundary conditions for 2D test:")
    print(variables.cell_vars[..., VI.RHOU])
    x = variables.cell_vars[..., VI.RHOV]
    context = BCApplicationContext(scale_factor=10)
    # manager.apply_boundary_on_single_var_direction(
    #     variables.cell_vars[..., VI.RHOU], direction=direction, context=context
    # )
    for side in [BdrySide.TOP, BdrySide.LEFT, BdrySide.RIGHT, BdrySide.BOTTOM]:
        manager.apply_extra(variables.cell_vars[..., VI.RHOU], side, context)
    # manager.apply_boundary_on_all_sides(variables.cell_vars)
    print(variables.cell_vars[..., VI.RHOU])


if __name__ == "__main__":
    boundary_manager_2d()
