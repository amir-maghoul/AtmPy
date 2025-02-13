# main.py
from atmpy.test_cases.traveling_vortex import TravelingVortexTestCase
from atmpy.boundary_conditions.boundary_manager import BoundaryConditionManager
from atmpy.solver.solver import Solver


def main():
    # Initialize test case
    test_case = TravelingVortexTestCase()

    # Initialize BoundaryConditionManager
    bc_manager = BoundaryConditionManager()
    bc_manager.setup_conditions(test_case.boundary_conditions)

    # Initialize Solver with parameters and BoundaryConditionManager
    solver = Solver(
        boundary_condition_manager=bc_manager, parameters=test_case.parameters
    )

    # Set initial conditions in Solver
    solver.set_initial_conditions(test_case.initial_conditions)

    # Run the solver
    solver.solve()


if __name__ == "__main__":
    main()
