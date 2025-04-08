"""This module handles solving the equation for the pressure variable including the Laplace/Poisson equation."""

from abc import ABC, abstractmethod
import numpy as np
import scipy as sp
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from atmpy.pressure_solver.discrete_operations import AbstractDiscreteOperator
    from atmpy.pressure_solver.linear_solvers import ILinearSolver


class AbstractPressureSolver(ABC):
    """Create different definitions of pressure solvers."""

    def __init__(
        self,
        discrete_operator: "AbstractDiscreteOperator",
        linear_solver: "ILinearSolver",
    ):
        self.discrete_operator: "AbstractDiscreteOperator" = discrete_operator
        self.linear_solver: "ILinearSolver" = linear_solver


class ClassicalPressureSolver(AbstractPressureSolver):
    """
    PressureSolver encapsulates the pressure correction procedure.
    It assembles the operator for the pressure correction (using, for example,
    a discrete Laplacian), builds the right–hand side from the divergence and other
    source terms, and then solves the resulting linear system via an injected linear solver.

    After obtaining the pressure correction, it updates the pressure (or p2-like)
    diagnostic in the Variables object. In a complete implementation the solver would
    also update ghost node values via boundary routines.
    """

    def __init__(
        self,
        discrete_operator: "AbstractDiscreteOperator",
        linear_solver: "ILinearSolver",
    ):
        super().__init__(discrete_operator, linear_solver)

    def build_operator(self, variables, grid, dt):
        """
        Assemble the pressure correction operator as a linear operator.
        This might be built using a multigrid, 9-point, or 27-point stencil, for example.

        For this simplified example we assume a 2D discrete Laplacian over the grid.
        """
        shape = (
            variables.rho.shape
        )  # assume the same shape for the diagnostic pressure field
        size = np.prod(shape)

        # Define the operator's action (a dummy Laplacian for illustration)
        def matvec(x):
            # x comes in as a flattened vector. Reshape to grid form.
            u = x.reshape(shape)
            laplacian = np.zeros_like(u)
            # A simple finite-difference Laplacian for interior points:
            laplacian[1:-1, 1:-1] = (
                u[:-2, 1:-1] - 2 * u[1:-1, 1:-1] + u[2:, 1:-1]
            ) / grid.dx**2 + (
                u[1:-1, :-2] - 2 * u[1:-1, 1:-1] + u[1:-1, 2:]
            ) / grid.dy**2
            # (In practice, ghost cells and boundary conditions are important.)
            return laplacian.ravel()

        A = sp.sparse.linalg.LinearOperator((size, size), matvec=matvec)
        return A

    def build_rhs(self, variables, grid, dt):
        """
        Build the right-hand side (RHS) for the pressure correction.
        Typically this uses the divergence of the momentum field along with
        other contributions (for instance, compressibility or additional source terms).

        In the legacy code, functions such as divergence_nodes and adjustments for
        compressibility are used. Here we provide a simplified placeholder version.
        """
        shape = variables.rho.shape
        rhs = np.zeros(shape)

        # For example, compute a simple discrete divergence based on momentum differences.
        # (Replace this with a proper discrete operator as needed.)
        rhs[1:-1, 1:-1] = (variables.rhou[2:, 1:-1] - variables.rhou[:-2, 1:-1]) / (
            2 * grid.dx
        ) + (variables.rhov[1:-1, 2:] - variables.rhov[1:-1, :-2]) / (2 * grid.dy)
        # Additional modifications (e.g. compressibility weighting) may be applied.

        # Flatten rhs for the linear solver.
        return rhs.ravel()

    def solve(self, variables, grid, dt):
        """
        Perform the pressure correction step:
          1. Build the operator A for the pressure correction.
          2. Compute the right-hand side vector from the momentum divergence and sources.
          3. Solve the linear system A * p_corr = rhs with the injected linear solver.
          4. Reshape and update the diagnostic pressure field (e.g., p2_nodes or variables.pressure).

        Boundary updates and ghost cell enforcement can be handled here or delegated.
        """
        A = self.build_operator(variables, grid, dt)
        rhs = self.build_rhs(variables, grid, dt)

        # Solve the system (using, e.g., BiCGStab)
        solution = self.linear_solver.solve(A, rhs)
        # Reshape the solution to the grid dimensions.
        pressure_correction = solution.reshape(variables.rho.shape)

        # Update the pressure diagnostic field.
        # In the legacy code, this update might look like:
        #   mpv.p2_nodes += weight * dp2n  (with dp2n computed via pressure derivative kernels)
        try:
            # If the field already exists, update it.
            variables.pressure += pressure_correction * dt
        except AttributeError:
            # Otherwise, create it as a new field.
            variables.pressure = pressure_correction * dt

        # One might update boundary ghost values here as well,
        # e.g., boundary_manager.set_ghostnodes_pressure(variables.pressure, node, user_data)
        print("Pressure solve complete and pressure field updated.")
