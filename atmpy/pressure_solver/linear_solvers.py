"""This module is a simple module to solve linear equations using different manual/scipy solvers."""

from abc import ABC, abstractmethod
import scipy as sp
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class ILinearSolver(ABC):
    @abstractmethod
    def solve(self, A: "np.ndarray", b: "np.ndarray", tol: float, max_iter: int):
        pass


class BiCGStabSolver(ILinearSolver):
    def solve(self, A: "np.ndarray", b: "np.ndarray", tol: float, max_iter: int):
        # ... Use sp.sparse.linalg.bicgstab ...
        x, info = sp.sparse.linalg.bicgstab(A, b, tol=tol, maxiter=max_iter)
        return x, info


class GMRESSolver(ILinearSolver):
    def solve(self, A: "np.ndarray", b: "np.ndarray", tol: float, max_iter: int):
        x, info = sp.sparse.linalg.gmres(A, b, tol=tol, maxiter=max_iter)
        return x, info
