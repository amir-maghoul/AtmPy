from atmpy.flux.limiters import *
import numpy as np

def test_van_leer_1d():
    a = np.random.rand(4, 3)*10
    b = np.random.rand(4, 3)*10
    expected = 2*a*b/(a+b)
    expected[a*b < 0] = 0
    result = van_leer(a, b)
    assert np.allclose(result, expected)

def test_van_leer_1d_constant():
    a = b = np.ones((4, 3))*10
    expected = np.ones((4, 3))*10
    result = van_leer(a, b)
    assert np.allclose(result, expected)

def test_van_leer_2d():
    a = np.random.rand(4, 3, 2)*10
    b = np.random.rand(4, 3, 2)*10
    expected = 2*a*b/(a+b)
    expected[a*b < 0] = 0
    result = van_leer(a, b)
    assert np.allclose(result, expected)

def test_van_leer_2d_constant():
    a = b = np.ones((4, 3, 2))*10
    expected = np.ones((4, 3, 2))*10
    result = van_leer(a, b)
    assert np.allclose(result, expected)

def test_van_leer_3d():
    a = np.random.rand(4, 3, 2, 2)*10
    b = np.random.rand(4, 3, 2, 2)*10
    expected = 2*a*b/(a+b)
    expected[a*b < 0] = 0
    assert np.allclose(van_leer(a, b), expected)

def test_van_leer_3d_constant():
    a = b = np.ones((4, 3, 2, 2))*10
    expected = np.ones((4, 3, 2, 2))*10
    assert np.allclose(van_leer(a, b), expected)

def test_van_leer_3d_negative():
    a = np.random.rand(4, 3, 2, 2)*(-1)
    b = np.random.rand(4, 3, 2, 2)*10
    expected = np.zeros((4, 3, 2, 2))
    assert np.allclose(van_leer(a, b), expected)