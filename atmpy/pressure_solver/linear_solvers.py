"""This module is a simple module to solve linear equations using different manual/scipy solvers."""

from abc import ABC, abstractmethod
import scipy as sp

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

########################################################################################################################
############################################ ITERATION COUNTERS ########################################################
########################################################################################################################
class IterationCounter:
    def __init__(self):
        self.count = 0
        self.residuals = [] # Optional: to store residual history

    def __call__(self, residual_norm): # For bicgstab, the callback receives the residual norm
        self.count += 1
        self.residuals.append(residual_norm)

########################################################################################################################
######################################### SOLVERS ######################################################################
########################################################################################################################
class ILinearSolver(ABC):
    @abstractmethod
    def solve(
        self,
        A: "np.ndarray",
        b: "np.ndarray",
        rtol: float,
        max_iter: int,
        M: "np.ndarray",
    ):
        pass


class BiCGStabSolver(ILinearSolver):
    def solve(
        self,
        A: "np.ndarray",
        b: "np.ndarray",
        rtol: float,
        max_iter: int,
        M: "np.ndarray",
    ):
        iterations = IterationCounter()
        x, info = sp.sparse.linalg.bicgstab(A, b, rtol=rtol, maxiter=max_iter, M=M, callback=iterations)
        self.iterations_ = iterations.count
        return x, info


class GMRESSolver(ILinearSolver):
    def solve(
        self,
        A: "np.ndarray",
        b: "np.ndarray",
        rtol: float,
        max_iter: int,
        M: "np.ndarray",
    ):
        iterations = IterationCounter()
        x, info = sp.sparse.linalg.gmres(A, b, rtol=rtol, maxiter=max_iter, M=M, callback=iterations)
        self.iterations_ = iterations.count
        return x, info
