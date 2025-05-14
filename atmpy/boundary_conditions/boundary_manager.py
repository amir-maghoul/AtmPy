from typing import Dict, List, Tuple, TYPE_CHECKING, Optional
import logging

if TYPE_CHECKING:
    from atmpy.boundary_conditions.boundary_conditions import (
        BaseBoundaryCondition as BaseBC,
    )
    import numpy as np
    from atmpy.boundary_conditions.bc_extra_operations import (
        ExtraBCOperation,
        PeriodicAdjustment,
    )
from atmpy.infrastructure.factory import get_boundary_conditions
from atmpy.infrastructure.enums import (
    BoundaryConditions as BdryType,
    BoundarySide as BdrySide,
)
from atmpy.infrastructure.utility import side_direction_mapping
from atmpy.boundary_conditions.contexts import (
    BoundaryConditionsConfiguration,
    BCApplicationContext,
)


class BoundaryManager:
    """The Boundary manager class to apply boundary condition as the strategy."""

    def __init__(self, bc_config: "BoundaryConditionsConfiguration") -> None:
        # The keys will be the boundary sides and the values are the created BC instances.
        self.boundary_conditions: Dict[BdrySide, "BaseBC"] = {}
        self.mpv_boundary_conditions: Dict[BdrySide, "BaseBC"] = {}
        self.setup_conditions(bc_config)

    def setup_conditions(self, bc_config: "BoundaryConditionsConfiguration") -> None:
        """Set up the boundary conditions in dictionaries for main and MPV variables."""
        self.boundary_conditions.clear()
        self.mpv_boundary_conditions.clear()

        for inst_opts in bc_config.options:
            # Validate side/direction compatibility
            BoundaryManager._validate_side_direction_compatibility(
                inst_opts.side, inst_opts.direction
            )

            ############################### Create and store PRIMARY BC instance #######################################
            bc_instance = get_boundary_conditions(inst_opts.type, inst_opts)
            self.boundary_conditions[inst_opts.side] = bc_instance

            ############################### Create and store MPV BC instance (if specified) ############################
            mpv_type_to_use = inst_opts.mpv_boundary_type
            if mpv_type_to_use is None:
                raise ValueError("The MPV boundary type must be specified.")

            mpv_bc_instance = get_boundary_conditions(
                inst_opts.mpv_boundary_type, inst_opts
            )
            self.mpv_boundary_conditions[inst_opts.side] = mpv_bc_instance

    @staticmethod
    def _validate_side_direction_compatibility(side: BdrySide, direction: str) -> None:
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
        # Contexts might need adjustment if BC apply signatures change
        # contexts: List["BCApplicationContext"] = [None], # Keep or adjust as needed
    ):
        """Apply the PRIMARY boundary conditions on a single side."""
        logging.debug(f"Apply boundary conditions *main variables* on side: {side}")
        if side not in self.boundary_conditions:  # Check primary dict
            raise ValueError(
                f"No primary boundary condition configured for side: {side}"
            )
        condition = self.boundary_conditions[side]
        # Assuming apply takes only 'cells' for now, add context if needed
        condition.apply(cells)  # Pass context[0] if required

    def apply_boundary_on_direction(
        self,
        cells: "np.ndarray",
        direction: str,
        # contexts: List["BCApplicationContext"] = [None, None], # Keep or adjust
    ):
        """Apply the PRIMARY boundary conditions on a single direction."""
        sides: Tuple[BdrySide, BdrySide] = side_direction_mapping(direction)
        logging.debug(f"Apply boundary conditions *main variables* on sides: {sides}")
        for side in sides:  # Add context iteration if needed
            if side in self.boundary_conditions:
                condition = self.boundary_conditions[side]
                condition.apply(cells)  # Pass context if required

    def apply_boundary_on_all_sides(self, cells: "np.ndarray"):
        """Apply the PRIMARY boundary conditions on all sides."""
        logging.debug(
            "Applying full PRIMARY boundary conditions on *main variables*..."
        )
        for side, condition in self.boundary_conditions.items():
            condition.apply(cells)  # Pass context if required

    def apply_boundary_on_single_var_one_side(
        self,
        variable: "np.ndarray",
        side: "BdrySide",
        contexts: List["BCApplicationContext"] = [None],
    ):
        """Apply the PRIMARY boundary for a single variable on one side."""
        logging.debug(f"Apply PRIMARY BC on single variable on the side: {side}")
        if side not in self.boundary_conditions:
            raise ValueError(
                f"No primary boundary condition configured for side: {side}"
            )
        condition = self.boundary_conditions[side]
        condition.apply_single_variable(variable, contexts[0])

    def apply_boundary_on_single_var_direction(
        self,
        variable: "np.ndarray",
        direction: str,
        contexts: List["BCApplicationContext"] = [None, None],
    ):
        """Apply the PRIMARY boundary for a single variable on a single direction."""
        sides = side_direction_mapping(direction)
        logging.debug(f"Apply PRIMARY BC on single variable on the sides: {sides}")
        # Corrected iteration: zip sides with contexts
        for side, context in zip(sides, contexts):
            if side in self.boundary_conditions:
                condition = self.boundary_conditions[side]
                condition.apply_single_variable(variable, context)

    def apply_boundary_on_single_var_all_sides(
        self, variable: "np.ndarray", contexts: List["BCApplicationContext"]
    ):
        """Apply the PRIMARY boundary conditions on all sides for the input variable."""
        logging.debug("Apply PRIMARY BC on single variable on all sides...")
        # Corrected iteration: Use boundary_conditions dict items
        if len(contexts) != len(self.boundary_conditions):
            raise ValueError(
                "Number of contexts must match number of boundary conditions"
            )
        i = 0
        for side, condition in self.boundary_conditions.items():
            condition.apply_single_variable(variable, contexts[i])
            i += 1

    def apply_pressure_boundary_on_one_side(
        self,
        mpv_vars: "np.ndarray",  # Assuming mpv vars are passed directly (e.g., mpv.p2_cells)
        side: BdrySide,
        context: Optional[
            "BCApplicationContext"
        ] = None,  # Context for this specific application
    ):
        """Apply the MPV-specific boundary condition on a single side."""
        logging.debug(f"Apply boundary conditions *MPV variables* on side: {side}")
        if side not in self.mpv_boundary_conditions:  # Check MPV dict
            raise ValueError(f"No MPV boundary condition configured for side: {side}")
        condition = self.mpv_boundary_conditions[side]
        if context is None:
            context = BCApplicationContext()
        condition.apply_single_variable(mpv_vars, context)

    def apply_pressure_boundary_on_direction(
        self,
        mpv_vars: "np.ndarray",
        direction: str,
        contexts: List[Optional["BCApplicationContext"]] = [None, None],
    ):
        """Apply the MPV-specific boundary conditions on a single direction."""
        sides: Tuple[BdrySide, BdrySide] = side_direction_mapping(direction)
        logging.debug(f"Apply boundary conditions *MPV variables* on sides: {sides}")
        for side, context in zip(sides, contexts):
            if side in self.mpv_boundary_conditions:
                condition = self.mpv_boundary_conditions[side]
                if context is None:
                    context = BCApplicationContext()
                condition.apply_single_variable(mpv_vars, context)

    def apply_pressure_boundary_on_all_sides(
        self,
        mpv_vars: "np.ndarray",
        contexts: Optional[List["BCApplicationContext"]] = None,
    ):
        """Apply the MPV-specific boundary conditions on all sides."""
        logging.debug(
            "Applying full MPV boundary conditions on *pressure variables*..."
        )
        num_bcs = len(self.mpv_boundary_conditions)
        if contexts is None:
            contexts = [BCApplicationContext() for _ in range(num_bcs)]
        elif len(contexts) != num_bcs:
            raise ValueError(
                "Number of contexts must match number of MPV boundary conditions"
            )

        i = 0
        for side, condition in self.mpv_boundary_conditions.items():
            condition.apply_single_variable(mpv_vars, contexts[i])
            i += 1

    def apply_extra(
        self,
        variable: "np.ndarray",
        side: "BdrySide",
        operations: List["ExtraBCOperation"],  # Should likely be single operation
        target_mpv: bool = False,
    ) -> None:
        """Apply extra conditions on the given single variable for a single side.
        Targets primary BC by default, use target_mpv=True for MPV BC."""

        target_dict = (
            self.mpv_boundary_conditions if target_mpv else self.boundary_conditions
        )
        bc_dict_name = "MPV" if target_mpv else "Primary"

        logging.debug(
            f"Apply EXTRA boundary conditions ({bc_dict_name}) on the side: {side}"
        )
        if side not in target_dict:
            raise ValueError(
                f"The side {side} does not exist in the {bc_dict_name} list."
            )

        condition = target_dict[side]
        # Assuming operation list contains only one relevant operation for the side
        if operations:
            # Check if the operation type matches the condition type?
            # The ExtraBCOperation base class already has target_type/target_side
            # Let's assume the caller provides the correct operation.
            condition.apply_extra(variable, operations[0])  # Pass the operation object

    def apply_extra_all_sides(
        self,
        variable: "np.ndarray",
        operations: List["ExtraBCOperation"],
        target_mpv: bool = False,  # --- NEW FLAG ---
    ) -> None:
        """Applies a batch of potentially targeted operations to primary or MPV BCs."""

        target_dict = (
            self.mpv_boundary_conditions if target_mpv else self.boundary_conditions
        )
        bc_dict_name = "MPV" if target_mpv else "Primary"

        if not operations:
            return
        logging.debug(
            f"\n--- Applying Batch of {len(operations)} Specific Operations ({bc_dict_name}) ---"
        )

        for operation in operations:
            # Handle operations targeting ALL sides
            if operation.target_side == BdrySide.ALL:
                applied = False
                for condition in target_dict.values():
                    # Check if the operation's target_type matches the condition's type
                    if (
                        operation.target_type is not None
                        or condition.type == operation.target_type
                    ):
                        condition.apply_extra(variable, operation)
                        applied = True
                if not applied:
                    print(
                        f"  Warning: Operation {operation.get_identifier()} did not match any {bc_dict_name} conditions."
                    )

            # Handle operations targeting specific sides
            else:
                target_side = operation.target_side
                logging.debug(
                    f"Applying extra BC ({bc_dict_name}) on side {target_side}"
                )
                condition = target_dict.get(target_side)
                if condition:
                    # Check if operation's target type matches condition's type
                    if (
                        operation.target_type is not None
                        or condition.type == operation.target_type
                    ):
                        condition.apply_extra(variable, operation)
                    else:
                        print(
                            f"  Skipped: Operation type {operation.target_type} mismatch for condition type {condition.type} on side {target_side}"
                        )

                else:
                    print(
                        f"  Skipped: Side {target_side} not found in {bc_dict_name} conditions."
                    )


def boundary_manager_2d_updated():
    from atmpy.physics.thermodynamics import Thermodynamics
    from atmpy.grid.utility import DimensionSpec, create_grid
    from atmpy.variables.variables import Variables
    from atmpy.variables.multiple_pressure_variables import MPV  # Import MPV
    from atmpy.infrastructure.enums import VariableIndices as VI
    from atmpy.boundary_conditions.bc_extra_operations import WallAdjustment
    from atmpy.boundary_conditions.contexts import (
        BCInstantiationOptions,
        BoundaryConditionsConfiguration,
        BCApplicationContext,
    )
    import numpy as np

    np.set_printoptions(linewidth=300)
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=10)

    # Grid setup (same as before)
    nx = 6
    ngx = 2
    ny = 5
    ngy = 2
    dim = [DimensionSpec(nx, 0, 2, ngx), DimensionSpec(ny, 0, 2, ngy)]
    grid = create_grid(dim)

    # Variable setup (same as before)
    variables = Variables(grid, 6, 1)
    variables.cell_vars.fill(1.0)  # Fill with baseline
    variables.cell_vars[grid.get_inner_slice() + (VI.RHO,)] = 5.0  # Inner rho = 5
    mpv = MPV(grid, num_vars=6, direction="y")  # Create MPV object
    mpv.p2_cells.fill(100.0)  # Example p2 values
    mpv.p2_cells[grid.get_inner_slice()] = np.arange(30).reshape(
        (nx, ny)
    )  # Inner p2 = 0

    # BC Configuration with MPV types
    stratification = lambda y: 1.0  # Example stratification
    th = Thermodynamics()
    gravity = (0.0, 1.0, 0.0)

    # Bottom: Reflective Gravity for main vars, WALL for MPV vars
    bc_bottom = BCInstantiationOptions(
        side=BdrySide.BOTTOM,
        type=BdryType.REFLECTIVE_GRAVITY,
        mpv_boundary_type=BdryType.WALL,
        direction="y",
        grid=grid,
        stratification=stratification,
        gravity=gravity,
    )
    # Top: Reflective Gravity for main vars, WALL for MPV vars
    bc_top = BCInstantiationOptions(
        side=BdrySide.TOP,
        type=BdryType.REFLECTIVE_GRAVITY,
        mpv_boundary_type=BdryType.WALL,
        direction="y",
        grid=grid,
        stratification=stratification,
        gravity=gravity,
    )
    # Left: Periodic for main vars, Periodic for MPV vars (MPV type could be None to default)
    bc_left = BCInstantiationOptions(
        side=BdrySide.LEFT,
        type=BdryType.WALL,
        mpv_boundary_type=BdryType.PERIODIC,
        direction="x",
        grid=grid,
    )
    # Right: Periodic for main vars, Periodic for MPV vars
    bc_right = BCInstantiationOptions(
        side=BdrySide.RIGHT,
        type=BdryType.WALL,
        # mpv_boundary_type=BdryType.PERIODIC,
        direction="x",
        grid=grid,
    )

    options = [bc_bottom, bc_top, bc_left, bc_right]
    print(bc_right.mpv_boundary_type)
    bc_config = BoundaryConditionsConfiguration(options)

    # Create Manager
    manager = BoundaryManager(bc_config)

    print("--- Initial State ---")
    print("Rho:\n", variables.cell_vars[..., VI.RHO])
    print("P2 Cells:\n", mpv.p2_cells)

    # Apply BCs
    print("\n--- Applying Boundaries ---")
    manager.apply_boundary_on_all_sides(variables.cell_vars)
    # Assuming MPV p2_cells are handled like single vars, use apply_mpv_...
    mpv.state(gravity, 0.115)
    manager.apply_pressure_boundary_on_all_sides(mpv.p2_cells)

    print("\n--- State After BC Application ---")
    print(
        "Rho (Reflective Gravity applied on Y boundaries):\n",
        variables.cell_vars[..., VI.RHO],
    )
    print("P2 Cells (WALL applied on Y boundaries, PERIODIC on X):\n", mpv.p2_cells)

    boundary_operation = [
        WallAdjustment(target_side=BdrySide.LEFT, target_type=BdryType.WALL, factor=100)
    ]
    manager.apply_extra_all_sides(
        variables.cell_vars[..., VI.RHO], boundary_operation, target_mpv=False
    )
    print(
        " RHO (EXTRA applied on Y boundaries, PERIODIC on X):\n",
        variables.cell_vars[..., VI.RHO],
    )


if __name__ == "__main__":
    boundary_manager_2d_updated()
