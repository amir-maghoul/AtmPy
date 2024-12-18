import pytest
import numpy as np
from unittest.mock import MagicMock
from atmpy.variables.variables import BaseVariableContainer, Variable, NodeVariable


# Mock variable indices for testing:
class MockVarInd1D:
    RHO = 0
    RHOU = 1
    P = 2
    PX = 3

class MockVarInd2D:
    RHO = 0
    RHOU = 1
    RHOV = 2
    P = 3
    PX = 4

class MockVarInd3D:
    RHO = 0
    RHOU = 1
    RHOV = 2
    RHOW = 3
    P = 4
    PX = 5

# Mocking a simple Grid class
class MockGrid:
    def __init__(self, dimensions, ncx, ncy=0, ncz=0):
        self.dimensions = dimensions
        self.ncx_total = ncx
        if dimensions >= 2:
            self.ncy_total = ncy
        if dimensions == 3:
            self.ncz_total = ncz

        # For NodeVariable
        self.nnx_total = ncx + 1
        if dimensions >= 2:
            self.nny_total = ncy + 1
        if dimensions == 3:
            self.nnz_total = ncz + 1

# Temporary modify sys.modules to replace atmpy.data.constants with
# MagicMock defined above to isolate the test
import sys
sys.modules['atmpy.data.constants'] = MagicMock(VarInd1D=MockVarInd1D,
                                                VarInd2D=MockVarInd2D,
                                                VarInd3D=MockVarInd3D)


@pytest.mark.parametrize("dims", [1, 2, 3])
def test_base_variable_container_init(dims):
    # Minimal dimension-based setup
    if dims == 1:
        g = MockGrid(dimensions=1, ncx=10)
    elif dims == 2:
        g = MockGrid(dimensions=2, ncx=10, ncy=20)
    elif dims == 3:
        g = MockGrid(dimensions=3, ncx=10, ncy=20, ncz=30)

    container = BaseVariableContainer(grid=g, num_vars=4)
    assert container.ndim == dims
    assert container.num_vars == 4
    assert container.grid is g

def test_base_variable_container_invalid_dims():
    g = MockGrid(dimensions=4, ncx=10)  # Not supported
    with pytest.raises(ValueError):
        BaseVariableContainer(grid=g, num_vars=4)

@pytest.mark.parametrize("dims", [1, 2, 3])
def test_variable_init_and_shape(dims):
    num_vars = 4
    if dims == 1:
        g = MockGrid(dimensions=1, ncx=10)
        var = Variable(g, num_vars)
        assert var.vars.shape == (10, num_vars)
    elif dims == 2:
        g = MockGrid(dimensions=2, ncx=10, ncy=20)
        var = Variable(g, num_vars)
        assert var.vars.shape == (10, 20, num_vars)
    elif dims == 3:
        g = MockGrid(dimensions=3, ncx=10, ncy=20, ncz=30)
        var = Variable(g, num_vars)
        assert var.vars.shape == (10, 20, 30, num_vars)

def test_variable_update_get():
    g = MockGrid(dimensions=1, ncx=5)
    var = Variable(g, 4)
    new_values = np.ones((5,4))
    var.update_vars(new_values)
    assert np.all(var.get_conservative_vars() == new_values)

@pytest.mark.parametrize("dims", [1, 2, 3])
def test_node_variable_init_and_shape(dims):
    num_vars = 4
    if dims == 1:
        g = MockGrid(dimensions=1, ncx=10)
        node_var = NodeVariable(g, num_vars)
        assert node_var.vars.shape == (11, num_vars)
    elif dims == 2:
        g = MockGrid(dimensions=2, ncx=10, ncy=20)
        node_var = NodeVariable(g, num_vars)
        assert node_var.vars.shape == (11, 21, num_vars)
    elif dims == 3:
        g = MockGrid(dimensions=3, ncx=10, ncy=20, ncz=30)
        node_var = NodeVariable(g, num_vars)
        assert node_var.vars.shape == (11, 21, 31, num_vars)

def test_node_variable_update_get():
    g = MockGrid(dimensions=2, ncx=5, ncy=5)
    node_var = NodeVariable(g, 3)
    new_values = np.ones((6,6,3))
    node_var.update_node_vars(new_values)
    assert np.all(node_var.get_node_vars() == new_values)

@pytest.mark.parametrize("dims", [1, 2, 3])
def test_to_primitive(dims):
    if dims == 1:
        g = MockGrid(dimensions=1, ncx=5)
        var = Variable(g, 4)  # [rho, rho*u, P, PX]
        var.vars[...,0] = 9999.0  # Unused index, filler
        var.vars[...,1] = 4.0     # rho*u
        var.vars[...,2] = 600.0   # P
        var.vars[...,3] = 2.0     # PX = rho
        prim = var.to_primitive() # [u, P, X]
        assert prim.shape == (5,3)
        assert np.allclose(prim[...,0], 0.0004, rtol=10e-5)
        assert np.allclose(prim[...,1], 600.0)
        assert np.allclose(prim[...,2], 2.0/600.0)



    elif dims == 2:
        g = MockGrid(dimensions=2, ncx=5, ncy=5)
        var = Variable(g, 5)  # [rho, rho*u, rho*v, P, PX]
        var.vars[...,0] = 9999.0
        var.vars[...,1] = 4.0   # rho*u
        var.vars[...,2] = 6.0   # rho*v
        var.vars[...,3] = 600.0 # P
        var.vars[...,4] = 2.0   # PX
        prim = var.to_primitive() # [u, v, P, X]
        assert prim.shape == (5,5,4)
        assert np.allclose(prim[...,0], 0.0004, rtol=10e-5)
        assert np.allclose(prim[...,1], 0.0006, rtol=10e-5)
        assert np.allclose(prim[...,2], 600)
        assert np.allclose(prim[...,3], 2/600)

    elif dims == 3:
        g = MockGrid(dimensions=3, ncx=5, ncy=5, ncz=5)
        var = Variable(g, 6)  # [rho, rho*u, rho*v, rho*w, P, PX]
        var.vars[...,0] = 9999.0    # rho
        var.vars[...,1] = 4.0       # rho*u
        var.vars[...,2] = 6.0       # rho*v
        var.vars[...,3] = 8.0       # rho*w
        var.vars[...,4] = 600.0     # P
        var.vars[...,5] = 2.0       # PX
        prim = var.to_primitive() # [u, v, w, P, X]
        assert prim.shape == (5,5,5,5)
        assert np.allclose(prim[...,0], 0.0004, rtol=10e-5)
        assert np.allclose(prim[...,1], 0.0006, rtol=10e-5)
        assert np.allclose(prim[...,2], 0.0008, rtol=10e-5)
        assert np.allclose(prim[...,3], 600)
        assert np.allclose(prim[...,4], 2/600)