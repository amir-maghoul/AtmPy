# tests/test_eos.py
import pytest
import numpy as np
from atmpy.physics.eos import (
    IdealGasEOS,
    BarotropicEOS,
    ExnerBasedEOS,
    P_to_pressure_numba,
    exner_to_pressure_numba,
)


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
    expected_p = K * rho**gamma
    computed_p = eos.pressure(rho)
    np.testing.assert_allclose(computed_p, expected_p, rtol=1e-6)


def test_barotropic_eos_sound_speed():
    K = 1.0
    gamma = 1.4
    eos = BarotropicEOS(K=K, gamma=gamma)
    rho = np.array([1.0, 2.0, 3.0])
    p = K * rho**gamma
    expected_a = np.sqrt(gamma * p / rho)
    computed_a = eos.sound_speed(rho, p)
    np.testing.assert_allclose(computed_a, expected_a, rtol=1e-6)


# Test cases for ExnerBasedEOS
def test_exner_based_eos_pressure_P_arr():
    cp = 1004.5  # J/(kg·K), specific heat at constant pressure for dry air
    cv = 718.2  # J/(kg·K), specific heat at constant volume for dry air
    R = cp - cv
    p_ref = 100000.0  # Reference pressure in Pascals
    eos = ExnerBasedEOS(cp=cp, cv=cv, p_ref=p_ref)

    P = np.array([1.0, 2.0, 3.0])
    # Expected p = p_ref * (P * R / p_ref)^(cp / cv)
    expected_p = p_ref * (P * R / p_ref) ** (cp / cv)
    computed_p = eos.pressure(P=P)
    np.testing.assert_allclose(computed_p, expected_p, rtol=1e-6)


def test_exner_based_eos_pressure_pi_arr():
    cp = 1004.5
    cv = 718.2
    R = cp - cv
    p_ref = 100000.0
    eos = ExnerBasedEOS(cp=cp, cv=cv, p_ref=p_ref)

    pi_arr = np.array([1.0, 1.1, 0.9])
    # Expected p = p_ref * pi_arr^(cp/R)
    expected_p = p_ref * pi_arr ** (cp / R)
    computed_p = eos.pressure(pi=pi_arr)
    np.testing.assert_allclose(computed_p, expected_p, rtol=1e-6)


def test_exner_based_eos_pressure_invalid():
    cp = 1004.5
    cv = 718.2
    R = cp - cv
    p_ref = 100000.0
    eos = ExnerBasedEOS(cp=cp, cv=cv, p_ref=p_ref)

    with pytest.raises(ValueError):
        eos.pressure()


def test_exner_based_eos_sound_speed():
    cp = 1004.5
    cv = 718.2
    p_ref = 100000.0
    eos = ExnerBasedEOS(cp=cp, cv=cv, p_ref=p_ref)

    rho_arr = np.array([1.0, 2.0, 4.0])
    p_arr = np.array([100000.0, 200000.0, 400000.0])
    gamma = cp / cv
    expected_a = np.sqrt(gamma * p_arr / rho_arr)
    computed_a = eos.sound_speed(rho_arr, p_arr)
    np.testing.assert_allclose(computed_a, expected_a, rtol=1e-6)


# Edge case tests
def test_eos_ideal_gas_zero_density():
    gamma = 1.4
    eos = IdealGasEOS(gamma=gamma)
    rho = np.array([0.0, 1.0])
    e = np.array([1.0, 2.0])
    # Pressure should be zero where density is zero
    expected_p = np.array([0.0, (gamma - 1.0) * 1.0 * 2.0])
    computed_p = eos.pressure(rho, e)
    np.testing.assert_allclose(computed_p, expected_p, rtol=1e-6)

    p = np.array([0.0, (gamma - 1.0) * 1.0 * 2.0])
    expected_a = np.array([0.0, np.sqrt(gamma * p[1] / rho[1])])
    computed_a = eos.sound_speed(rho, p)
    np.testing.assert_allclose(computed_a, expected_a, rtol=1e-6)


def test_barotropic_eos_zero_density():
    K = 1.0
    gamma = 1.4
    eos = BarotropicEOS(K=K, gamma=gamma)
    rho = np.array([0.0, 2.0])
    # Pressure should be zero where density is zero
    expected_p = np.array([0.0, K * 2.0**gamma])
    computed_p = eos.pressure(rho)
    np.testing.assert_allclose(computed_p, expected_p, rtol=1e-6)

    p = expected_p
    expected_a = np.sqrt(gamma * p / rho)
    # Handle division by zero for density = 0
    with np.errstate(divide="ignore", invalid="ignore"):
        computed_a = eos.sound_speed(rho, p)
    expected_a[0] = 0.0  # Define sound speed as zero where density is zero
    np.testing.assert_allclose(computed_a, expected_a, rtol=1e-6)


def test_exner_based_eos_zero_density_sound_speed():
    """
    Test the ExnerBasedEOS.sound_speed method to ensure that sound speed is zero
    where density is zero, preventing division by zero and resulting NaN values.
    """
    # Define EOS parameters (example values)
    cp = 1004.5  # J/(kg·K), specific heat at constant pressure for dry air
    cv = 718.2  # J/(kg·K), specific heat at constant volume for dry air
    R = cp - cv
    p_ref = 100000.0  # Reference pressure in Pascals

    # Instantiate the ExnerBasedEOS
    eos = ExnerBasedEOS(cp=cp, cv=cv, p_ref=p_ref)

    # Define test input arrays with zero density
    rho_arr = np.array([0.0, 1.0, 0.0, 2.0, 3.0, 0.0])
    p_arr = np.array([0.0, 100000.0, 0.0, 200000.0, 300000.0, 0.0])

    # Compute expected sound speeds
    gamma = cp / cv
    expected_a = np.sqrt(gamma * p_arr / rho_arr)
    # Handle cases where rho_arr == 0 by setting sound speed to 0
    expected_a = np.where(rho_arr == 0, 0.0, expected_a)

    # Compute sound speeds using the ExnerBasedEOS
    computed_a = eos.sound_speed(rho_arr, p_arr)

    # Assert that computed sound speeds match expected values
    np.testing.assert_allclose(computed_a, expected_a, rtol=1e-6)

    # Additionally, verify that sound speed is zero where rho_arr is zero
    assert np.all(
        computed_a[rho_arr == 0] == 0.0
    ), "Sound speed should be 0 where density is zero."


def test_exner_based_eos_negative_pressure():
    cp = 1004.5
    cv = 718.2
    p_ref = 100000.0
    eos = ExnerBasedEOS(cp=cp, cv=cv, p_ref=p_ref)

    P_arr = np.array([-1.0, 2.0])  # Negative P is physically invalid
    with pytest.raises(ValueError):
        eos.pressure(P=P_arr)


def test_exner_based_eos_negative_pi():
    cp = 1004.5
    cv = 718.2
    p_ref = 100000.0
    eos = ExnerBasedEOS(cp=cp, cv=cv, p_ref=p_ref)

    pi_arr = np.array([1.0, -1.0])  # Negative pi is physically invalid
    with pytest.raises(ValueError):
        eos.pressure(pi=pi_arr)
