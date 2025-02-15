# main.py
from atmpy.test_cases.traveling_vortex import TravelingVortexTestCase
from atmpy.boundary_conditions.boundary_manager import BoundaryConditionManager
from atmpy.solver.solver import Solver


def main():
    # Initialize the specific test case
    test_case = TravelingVortexTestCase()

    # Access the updated simulation configuration
    simulation_config = test_case.config

    # Initialize the solver with the configuration
    solver = Solver(simulation_config)

    # Run the simulation
    solver.run()



if __name__ == "__main__":
    main()
