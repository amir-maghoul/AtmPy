import numpy as np
from numba import njit, prange
from atmpy.variables.variables import Variables
from atmpy.physics.eos import *
from atmpy.data.enums import *


def roe(left_state: Variables, right_state: Variables, direction: str):
    raise NotImplementedError(f"Roe solver for {direction}-direction not implemented.")


@njit
def hll(left_state: Variables, right_state: Variables, direction: str):
    raise NotImplementedError(f"HLL solver for {direction}-direction not implemented.")


def hllc(left_state: Variables, right_state: Variables, direction: str):
    raise NotImplementedError(f"HLLC solver for {direction}-direction not implemented.")


def rusanov(left_state: Variables, right_state: Variables, direction: str):
    raise NotImplementedError(
        f"Rusanov solver for {direction}-direction not implemented."
    )
