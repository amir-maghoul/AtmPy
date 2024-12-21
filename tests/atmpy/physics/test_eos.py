# tests/test_eos.py
import pytest
import numpy as np
from atmpy.physics.eos import IdealGasEOS, BarotropicEOS, ExnerBasedEOS, P_to_pressure_numba, exner_to_pressure_numba


# Test cases for IdealGasEOS
def test_ideal_gas_eos_pressure():
    gamma = 1.4
    eos = IdealGasEOS(gamma=gamma)
    rho = np.array([1.0, 2.0, 3.0])
    e = np.array([2.0, 3.0, 4.0])
    expected_p = (gamma - 1.0) * rho * e
    computed_p = eos.pressure(rho, e)
    np.testing.assert_allclose(computed_p, expected_p, rtol=1e-6)


def test_ideal_gas_eos_sound_speed():
    gamma = 1.4
    eos = IdealGasEOS(gamma=gamma)
    rho = np.array([1.0, 2.0, 4.0])
    p = np.array([1.4, 2.8, 5.6])
    expected_a = np.sqrt(gamma * p / rho)
    computed_a = eos.sound_speed(rho, p)
    np.testing.assert_allclose(computed_a, expected_a, rtol=1e-6)


# Test cases for BarotropicEOS
def test_barotropic_eos_pressure():
    K = 1.0
    gamma = 1.4
    eos = BarotropicEOS(K=K, gamma=gamma)
    rho = np.array([1.0, 2.0, 3.0, 4.0])
    expected_p = K * rho ** gamma
    computed_p = eos.pressure(rho)
    np.testing.assert_allclose(computed_p, expected_p, rtol=1e-6)


def test_barotropic_eos_sound_speed():
    K = 1.0
    gamma = 1.4
    eos = BarotropicEOS(K=K, gamma=gamma)
    rho = np.array([1.0, 2.0, 3.0])
    p = K * rho ** gamma
    expected_a = np.sqrt(gamma * p / rho)
    computed_a = eos.sound_speed(rho, p)
    np.testing.assert_allclose(computed_a, expected_a, rtol=1e-6)


# Test cases for ExnerBasedEOS
def test_exner_based_eos_pressure_P_arr():
    R = 287.05  # J/(kg·K), specific gas constant for dry air
    cp = 1004.5  # J/(kg·K), specific heat at constant pressure for dry air
    cv = 718.2  # J/(kg·K), specific heat at constant volume for dry air
    p_ref = 100000.0  # Reference pressure in Pascals
    eos = ExnerBasedEOS(R=R, cp=cp, cv=cv, p_ref=p_ref)

    P = np.array([1.0, 2.0, 3.0])
    # Expected p = p_ref * (P * R / p_ref)^(cp / cv)
    expected_p = p_ref * (P * R / p_ref) ** (cp / cv)
    computed_p = eos.pressure(P=P)
    np.testing.assert_allclose(computed_p, expected_p, rtol=1e-6)


def test_exner_based_eos_pressure_pi_arr():
    R = 287.05
    cp = 1004.5
    cv = 718.2
    p_ref = 100000.0
    eos = ExnerBasedEOS(R=R, cp=cp, cv=cv, p_ref=p_ref)

    pi_arr = np.array([1.0, 1.1, 0.9])
    # Expected p = p_ref * pi_arr^(cp/R)
    expected_p = p_ref * pi_arr ** (cp / R)
    computed_p = eos.pressure(pi=pi_arr)
    np.testing.assert_allclose(computed_p, expected_p, rtol=1e-6)


def test_exner_based_eos_pressure_invalid():
    R = 287.05
    cp = 1004.5
    cv = 718.2
    p_ref = 100000.0
    eos = ExnerBasedEOS(R=R, cp=cp, cv=cv, p_ref=p_ref)

    with pytest.raises(ValueError):
        eos.pressure()


def test_exner_based_eos_sound_speed():
    R = 287.05
    cp = 1004.5
    cv = 718.2
    p_ref = 100000.0
    eos = ExnerBasedEOS(R=R, cp=cp, cv=cv, p_ref=p_ref)

    rho_arr = np.array([1.0, 2.0, 4.0])
    p_arr = np.array([100000.0, 200000.0, 400000.0])
    gamma = cp / cv
    expected_a = np.sqrt(gamma * p_arr / rho_arr)
    computed_a = eos.sound_speed(rho_arr, p_arr)
    np.testing.assert_allclose(computed_a, expected_a, rtol=1e-6)
