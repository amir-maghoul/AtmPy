# Test utility functions directly
import pytest
from atmpy.physics.utility import *
from atmpy.physics.eos import *

def test_exner_to_pressure_numba():
    pi_arr = np.array([1.0, 1.2, 0.8])
    p_ref = 100000.0
    cp = 1004.5
    R = 287.05
    expected_p = p_ref * pi_arr ** (cp / R)
    computed_p = exner_to_pressure_numba(pi_arr, p_ref, cp, R)
    np.testing.assert_allclose(computed_p, expected_p, rtol=1e-6)


def test_P_to_pressure_numba():
    P_arr = np.array([1.0, 1.5, 2.0])
    p_ref = 100000.0
    cp = 1004.5
    cv = 718.2
    R = cp - cv
    fac = P_arr * R / p_ref
    expected_p = p_ref * (fac ** (cp / cv))
    computed_p = P_to_pressure_numba(P_arr, p_ref, cp, cv)
    np.testing.assert_allclose(computed_p, expected_p, rtol=1e-6)
