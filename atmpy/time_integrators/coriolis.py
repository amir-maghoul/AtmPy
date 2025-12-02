"""This module is responsible for calculation on the coriolis operator"""

import logging
from typing import TYPE_CHECKING, Union, Any, Tuple, Callable
import numpy as np

if TYPE_CHECKING:
    from atmpy.physics.gravity import Gravity
    from atmpy.variables.variables import Variables
    from atmpy.variables.multiple_pressure_variables import MPV
from atmpy.infrastructure.enums import VariableIndices as VI
from atmpy.time_integrators.coriolis_numba_kernels import apply_coriolis_transform_nb_y
from enum import Enum, auto

class CoriolisUse(Enum):
    """How the extended Coriolis inverse is being used."""
    MOMENTUM_UPDATE = auto()        # momentum state update
    HELMHOLTZ_OPERATOR = auto()    # building Helmholtz Operator


class CoriolisOperator:
    """
    Encapsulates the handling of coriolis operator in the implicit time update. This is a part of operator splitting
    for stiff coriolis operator.
    """

    # flag to avoid redundant precompilation of kernels
    _kernel_compiled = False

    def __init__(
        self,
        coriolis_strength: Union[np.ndarray, list, tuple],
        gravity: "Gravity",
    ):
        self.strength: np.ndarray = np.array(coriolis_strength)
        self.gravity: "Gravity" = gravity
        self.coriolis_inverse_kernel: Callable
        self._select_coriolis_kernel()
        self._precompile_numba()

    def _select_coriolis_kernel(self) -> None:
        if self.gravity.direction == "y":
            self.coriolis_inverse_kernel = apply_coriolis_transform_nb_y
        else:
            raise NotImplementedError(
                "The numba kernel for gravity directions other than y are not implemented."
            )

    def _precompile_numba(self):
        """Attempt to pre-compile the core Numba function."""
        if CoriolisOperator._kernel_compiled:
            logging.debug("Numba Coriolis kernel already compiled/attempted.")
            return
        try:
            print("Pre-compiling Numba Coriolis kernel...")
            # Create dummy data with minimal size but correct types/shapes
            dummy_shape = (2, 2, 2)  # Minimal 3D cell shape
            dummy_momentum = np.zeros(dummy_shape, dtype=np.float64)
            dummy_dchi = np.zeros(dummy_shape, dtype=np.float64)
            dummy_strength = np.zeros(3, dtype=np.float64)  # Coriolis strength shape

            # Call with dummy data
            _ = self.coriolis_inverse_kernel(
                dummy_momentum.copy(),  # Pass copies if modified in place
                dummy_momentum.copy(),
                dummy_momentum.copy(),
                dummy_dchi,
                True,  # nonhydro
                True,  # nongeo
                0.1,  # dt
                dummy_strength,
            )
            CoriolisOperator._kernel_compiled = True
            print("Numba Coriolis kernel pre-compiled successfully.")
        except Exception as e:
            print(f"\nWARNING: Could not pre-compile Numba Coriolis kernel.")
            print(f"Error during pre-compilation: {e}")
            print(f"Execution might be slower on the first call.\n")

    def apply_inverse_for_momentum_state(
            self,
            U: np.ndarray,
            V: np.ndarray,
            W: np.ndarray,
            variables: "Variables",
            mpv: "MPV",
            is_nongeostrophic: bool,
            is_nonhydrostatic: bool,
            Msq: float,
            dt: float,
    ) -> None:
        """M^{-1} acting on the *current momenta*, used in backward_explicit."""
        self._apply_inverse_core(
            U, V, W,
            variables, mpv,
            is_nongeostrophic,
            is_nonhydrostatic,
            Msq,
            dt,
            use=CoriolisUse.MOMENTUM_UPDATE,
        )

    def apply_inverse_for_helmholtz_operator(
            self,
            Pu: np.ndarray,
            Pv: np.ndarray,
            Pw: np.ndarray,
            variables: "Variables",
            mpv: "MPV",
            is_nongeostrophic: bool,
            is_nonhydrostatic: bool,
            Msq: float,
            dt: float,
    ) -> None:
        """M^{-1} acting on dt * (PΘ) ∇p, used for building the Helmholtz RHS."""
        self._apply_inverse_core(
            Pu, Pv, Pw,
            variables, mpv,
            is_nongeostrophic,
            is_nonhydrostatic,
            Msq,
            dt,
            use=CoriolisUse.HELMHOLTZ_OPERATOR,
        )

    def _apply_inverse_core(
            self,
            U: np.ndarray,
            V: np.ndarray,
            W: np.ndarray,
            variables: "Variables",
            mpv: "MPV",
            is_nongeostrophic: bool,
            is_nonhydrostatic: bool,
            Msq: float,
            dt: float,
            use: CoriolisUse,
    ) -> None:
        """
        The only place where we actually call the numba kernel.

        - Decides how to use the physical switches (is_nongeostrophic, is_nonhydrostatic)
        - Decides whether this is a 'pressure call' (is_pressure_calling)
        - Documents what each combination means.
        """
        nonhydro = is_nonhydrostatic
        nongeo = is_nongeostrophic

        g: float = self.gravity.strength
        dChi = mpv.compute_dS_on_nodes()
        Theta = variables.cell_vars[..., VI.RHOY] / variables.cell_vars[..., VI.RHO]
        dt_N_sq = (dt ** 2) * (g / Msq) * dChi * Theta

        # Algorithmic mapping: decide when the kernel should treat this as a
        # 'pressure-related' transform.
        if use is CoriolisUse.MOMENTUM_UPDATE:
            is_pressure_calling = is_nongeostrophic  # "pure" state update
        elif use is CoriolisUse.HELMHOLTZ_OPERATOR:
            is_pressure_calling = True  # building Helmholtz operator
        else:
            raise ValueError(f"Unsupported CoriolisUse: {use}")

        u_new, v_new, w_new = self.coriolis_inverse_kernel(
            U, V, W,
            dt_N_sq,
            bool(nonhydro),
            bool(nongeo),
            float(dt),
            self.strength,
            is_pressure_calling,
        )

        U[...] = u_new
        V[...] = v_new
        W[...] = w_new