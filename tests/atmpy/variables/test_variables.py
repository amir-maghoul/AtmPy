import pytest
import numpy as np
from unittest.mock import MagicMock
from atmpy.variables.variables import Variables  # Updated import statement

# Mock variable indices for testing:
class MockVariableIndices:
    RHO = 0
    RHOX = 1
    RHOY = 2
    RHOU = 3
    RHOV = 4
    RHOW = 5

    @staticmethod
    def values():
        return (
            MockVariableIndices.RHO,
            MockVariableIndices.RHOX,
            MockVariableIndices.RHOY,
            MockVariableIndices.RHOU,
            MockVariableIndices.RHOV,
            MockVariableIndices.RHOW,
        )

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

# Temporarily modify sys.modules to replace atmpy.data.constants with
# MagicMock defined above to isolate the test
import sys

sys.modules["atmpy.data.constants"] = MagicMock(
    VariableIndices=MockVariableIndices
)

@pytest.mark.parametrize("dims, num_cell_vars, num_node_vars", [
    (1, 4, 1),
    (2, 5, 2),
    (3, 6, 3)
])
def test_variables_init(dims, num_cell_vars, num_node_vars):
    """
    Test the initialization of the Variables class for different dimensions.
    """
    if dims == 1:
        g = MockGrid(dimensions=1, ncx=10)
    elif dims == 2:
        g = MockGrid(dimensions=2, ncx=10, ncy=20)
    elif dims == 3:
        g = MockGrid(dimensions=3, ncx=10, ncy=20, ncz=30)

    vars_container = Variables(grid=g, num_vars_cell=num_cell_vars, num_vars_node=num_node_vars)
    assert vars_container.ndim == dims
    assert vars_container.num_vars_cell == num_cell_vars
    assert vars_container.num_vars_node == num_node_vars
    assert vars_container.grid is g

    # Check shapes of cell_vars and primitives
    if num_cell_vars is not None:
        if dims == 1:
            expected_cell_shape = (g.ncx_total, num_cell_vars)
        elif dims == 2:
            expected_cell_shape = (g.ncx_total, g.ncy_total, num_cell_vars)
        elif dims == 3:
            expected_cell_shape = (g.ncx_total, g.ncy_total, g.ncz_total, num_cell_vars)
        assert vars_container.cell_vars.shape == expected_cell_shape

        # Check shape of primitives
        assert vars_container.primitives.shape == expected_cell_shape

    # Check shapes of node_vars
    if num_node_vars is not None:
        if dims == 1:
            expected_node_shape = (g.nnx_total, num_node_vars)
        elif dims == 2:
            expected_node_shape = (g.nnx_total, g.nny_total, num_node_vars)
        elif dims == 3:
            expected_node_shape = (g.nnx_total, g.nny_total, g.nnz_total, num_node_vars)
        assert vars_container.node_vars.shape == expected_node_shape

def test_variables_init_invalid_dims():
    """
    Test that Variables class raises a ValueError for unsupported dimensions.
    """
    g = MockGrid(dimensions=4, ncx=10)  # Not supported
    with pytest.raises(ValueError):
        Variables(grid=g, num_vars_cell=4, num_vars_node=1)

@pytest.mark.parametrize("dims, num_cell_vars, num_node_vars", [
    (1, 4, 2),
    (2, 5, 3),
    (3, 6, 4)
])
def test_variables_update_and_get_cell_vars(dims, num_cell_vars, num_node_vars):
    """
    Test updating and retrieving cell-centered variables.
    """
    if dims == 1:
        g = MockGrid(dimensions=1, ncx=5)
        expected_shape = (5, num_cell_vars)
    elif dims == 2:
        g = MockGrid(dimensions=2, ncx=5, ncy=5)
        expected_shape = (5, 5, num_cell_vars)
    elif dims == 3:
        g = MockGrid(dimensions=3, ncx=5, ncy=5, ncz=5)
        expected_shape = (5, 5, 5, num_cell_vars)

    vars_container = Variables(grid=g, num_vars_cell=num_cell_vars, num_vars_node=num_node_vars)

    # Create new cell data
    new_cell_data = np.ones(expected_shape) * 42.0  # Arbitrary test value
    vars_container.update_cell_vars(new_cell_data)
    retrieved_cell_data = vars_container.get_cell_vars()
    assert np.all(retrieved_cell_data == new_cell_data)

    # Ensure primitives are unchanged until to_primitive is called
    assert np.all(vars_container.primitives == 0.0)

@pytest.mark.parametrize("dims, num_cell_vars, num_node_vars", [
    (1, 3, 2),
    (2, 4, 3),
    (3, 5, 4)
])
def test_variables_update_and_get_node_vars(dims, num_cell_vars, num_node_vars):
    """
    Test updating and retrieving node-based variables.
    """
    if dims == 1:
        g = MockGrid(dimensions=1, ncx=4)
        expected_shape = (5, num_node_vars)
    elif dims == 2:
        g = MockGrid(dimensions=2, ncx=4, ncy=4)
        expected_shape = (5, 5, num_node_vars)
    elif dims == 3:
        g = MockGrid(dimensions=3, ncx=4, ncy=4, ncz=4)
        expected_shape = (5, 5, 5, num_node_vars)

    vars_container = Variables(grid=g, num_vars_cell=num_cell_vars, num_vars_node=num_node_vars)

    # Create new node data
    new_node_data = np.ones(expected_shape) * 24.0  # Arbitrary test value
    vars_container.update_node_vars(new_node_data)
    retrieved_node_data = vars_container.get_node_vars()
    assert np.all(retrieved_node_data == new_node_data)

@pytest.mark.parametrize("dims, num_cell_vars, num_node_vars, gamma", [
    (1, 4, 1, 1.4),
    (2, 5, 2, 1.4),
    (3, 6, 3, 1.4)
])
def test_variables_to_primitive(dims, num_cell_vars, num_node_vars, gamma):
    """
    Test the to_primitive method for converting conservative to primitive variables.
    """
    if dims == 1:
        g = MockGrid(dimensions=1, ncx=5)
        expected_cell_shape = (5, num_cell_vars)
    elif dims == 2:
        g = MockGrid(dimensions=2, ncx=5, ncy=5)
        expected_cell_shape = (5, 5, num_cell_vars)
    elif dims == 3:
        g = MockGrid(dimensions=3, ncx=5, ncy=5, ncz=5)
        expected_cell_shape = (5, 5, 5, num_cell_vars)

    vars_container = Variables(grid=g, num_vars_cell=num_cell_vars, num_vars_node=num_node_vars)

    # Populate cell_vars with test data
    # Set rho, rhoX, rhoY, rho*u, rho*v, rho*w as per the variable ordering
    # rho = 2.0, rhoX = 3.0, rhoY = 2.0, rho*u = 4.0, rho*v = 6.0, rho*w = 8.0

    # Initialize all variables to zero first to avoid residuals
    vars_container.cell_vars.fill(0.0)

    # Set rho
    vars_container.cell_vars[..., MockVariableIndices.RHO] = 2.0  # rho = 2.0

    # Set rhoX and rhoY
    vars_container.cell_vars[..., MockVariableIndices.RHOX] = 3.0  # rhoX = 3.0
    vars_container.cell_vars[..., MockVariableIndices.RHOY] = 2.0  # rhoY = 2.0


    # Set momentum variables
    vars_container.cell_vars[..., MockVariableIndices.RHOU] = 4.0  # rho*u = 4.0 => u = 2.0
    if dims >= 2:
        vars_container.cell_vars[..., MockVariableIndices.RHOV] = 6.0  # rho*v = 6.0 => v = 3.0
    if dims == 3:
        vars_container.cell_vars[..., MockVariableIndices.RHOW] = 8.0  # rho*w = 8.0 => w = 4.0

    # Invoke to_primitive
    vars_container.to_primitive(gamma=gamma)

    # Retrieve primitives
    prim = vars_container.primitives

    # Calculate expected primitives
    expected_p = 2.0 ** gamma  # p = rhoY ** gamma = 2.0 ** 1.4 ≈ 2.639
    expected_X_over_rho = 3.0 / 2.0  # 1.5
    expected_Y_over_rho = 2.0 / 2.0  # 1.0
    expected_u = 4.0 / 2.0  # 2.0
    if dims >= 2:
        expected_v = 6.0 / 2.0  # 3.0
    if dims == 3:
        expected_w = 8.0 / 2.0  # 4.0

    # Assertions
    if dims == 1:
        # primitives[...,0] = p
        # primitives[...,1] = X / rho
        # primitives[...,2] = Y / rho
        # primitives[...,3] = u
        assert prim.shape == expected_cell_shape
        assert np.allclose(prim[..., 0], expected_p, atol=1e-6)
        assert np.allclose(prim[..., 1], expected_X_over_rho, atol=1e-6)
        assert np.allclose(prim[..., 2], expected_Y_over_rho, atol=1e-6)
        assert np.allclose(prim[..., 3], expected_u, atol=1e-6)
    elif dims == 2:
        # primitives[...,0] = p
        # primitives[...,1] = X / rho
        # primitives[...,2] = Y / rho
        # primitives[...,3] = u
        # primitives[...,4] = v
        assert prim.shape == expected_cell_shape
        assert np.allclose(prim[..., 0], expected_p, atol=1e-6)
        assert np.allclose(prim[..., 1], expected_X_over_rho, atol=1e-6)
        assert np.allclose(prim[..., 2], expected_Y_over_rho, atol=1e-6)
        assert np.allclose(prim[..., 3], expected_u, atol=1e-6)
        assert np.allclose(prim[..., 4], expected_v, atol=1e-6)
    elif dims == 3:
        # primitives[...,0] = p
        # primitives[...,1] = X / rho
        # primitives[...,2] = Y / rho
        # primitives[...,3] = u
        # primitives[...,4] = v
        # primitives[...,5] = w
        assert prim.shape == expected_cell_shape
        assert np.allclose(prim[..., 0], expected_p, atol=1e-6)
        assert np.allclose(prim[..., 1], expected_X_over_rho, atol=1e-6)
        assert np.allclose(prim[..., 2], expected_Y_over_rho, atol=1e-6)
        assert np.allclose(prim[..., 3], expected_u, atol=1e-6)
        assert np.allclose(prim[..., 4], expected_v, atol=1e-6)
        assert np.allclose(prim[..., 5], expected_w, atol=1e-6)

def test_variables_print_debug_info(capfd):
    """
    Test the print_debug_info method of the Variables class.
    """
    num_cell_vars = 3
    num_node_vars = 2
    g = MockGrid(dimensions=2, ncx=10, ncy=20)
    vars_container = Variables(grid=g, num_vars_cell=num_cell_vars, num_vars_node=num_node_vars)

    # Call the debug method
    vars_container.print_debug_info()

    # Capture the output
    captured = capfd.readouterr()
    assert "Variables Info:" in captured.out
    assert "Dimensions: 2" in captured.out
    assert "Number of cell vars: 3" in captured.out
    assert "Number of node vars: 2" in captured.out
    assert f"Shape of cell_vars: {vars_container.cell_vars.shape}" in captured.out
    assert f"Shape of primitives: {vars_container.primitives.shape}" in captured.out
    assert f"Shape of node_vars: {vars_container.node_vars.shape}" in captured.out

def test_variables_no_cell_vars():
    """
    Test Variables class initialization when cell variables are not provided.
    """
    num_cell_vars = None
    num_node_vars = 2

    g = MockGrid(dimensions=2, ncx=4, ncy=4)
    with pytest.raises(ValueError):
        Variables(grid=g, num_vars_cell=num_cell_vars, num_vars_node=num_node_vars)

    num_cell_vars = 0
    with pytest.raises(ValueError):
        Variables(grid=g, num_vars_cell=num_cell_vars, num_vars_node=num_node_vars)

def test_variables_negative_cell_vars():
    """
    Test Variables class initialization when cell variables are not provided.
    """
    num_cell_vars = -1
    num_node_vars = 2

    g = MockGrid(dimensions=2, ncx=4, ncy=4)
    with pytest.raises(ValueError):
        Variables(grid=g, num_vars_cell=num_cell_vars, num_vars_node=num_node_vars)

def test_variables_no_node_vars():
    """
    Test Variables class initialization when cell variables are not provided.
    """
    num_cell_vars = 2
    num_node_vars = None

    g = MockGrid(dimensions=2, ncx=4, ncy=4)
    with pytest.raises(ValueError):
        Variables(grid=g, num_vars_cell=num_cell_vars, num_vars_node=num_node_vars)

    num_node_vars = 0
    with pytest.raises(ValueError):
        Variables(grid=g, num_vars_cell=num_cell_vars, num_vars_node=num_node_vars)

def test_variables_negative_node_vars():
    """
    Test Variables class initialization when cell variables are not provided.
    """
    num_cell_vars = 2
    num_node_vars = -1

    g = MockGrid(dimensions=2, ncx=4, ncy=4)
    with pytest.raises(ValueError):
        Variables(grid=g, num_vars_cell=num_cell_vars, num_vars_node=num_node_vars)

def test_variables_partial_update():
    """
    Test updating only cell_vars or node_vars without affecting the other.
    """
    num_cell_vars = 6  # Updated to match the number of variables for 3D
    num_node_vars = 2

    g = MockGrid(dimensions=3, ncx=3, ncy=3, ncz=3)
    vars_container = Variables(grid=g, num_vars_cell=num_cell_vars, num_vars_node=num_node_vars)

    # Update node_vars
    new_node_data = np.full((4, 4, 4, num_node_vars), 5.0)
    vars_container.update_node_vars(new_node_data)
    assert np.all(vars_container.get_node_vars() == new_node_data)

    # Ensure cell_vars remain unchanged (still zeros)
    assert np.all(vars_container.get_cell_vars() == 0.0)

    # Ensure primitives remain unchanged (still zeros)
    assert np.all(vars_container.primitives == 0.0)

    # Now update cell_vars with meaningful data
    # Set rho, rhoX, rhoY, rho*u, rho*v, rho*w
    vars_container.cell_vars[..., MockVariableIndices.RHO] = 2.0
    vars_container.cell_vars[..., MockVariableIndices.RHOX] = 3.0
    vars_container.cell_vars[..., MockVariableIndices.RHOY] = 2.0
    vars_container.cell_vars[..., MockVariableIndices.RHOU] = 4.0
    vars_container.cell_vars[..., MockVariableIndices.RHOV] = 6.0
    vars_container.cell_vars[..., MockVariableIndices.RHOW] = 8.0

    # Ensure node_vars remain unchanged
    assert np.all(vars_container.get_node_vars() == new_node_data)

    # Ensure cell_vars are updated correctly
    expected_cell_data = vars_container.cell_vars.copy()
    retrieved_cell_data = vars_container.get_cell_vars()
    assert np.all(retrieved_cell_data == expected_cell_data)

    # Ensure primitives are unchanged until to_primitive is called
    assert np.all(vars_container.primitives == 0.0)

    # Now call to_primitive and ensure primitives are updated correctly
    gamma = 1.4
    vars_container.to_primitive(gamma=gamma)
    prim = vars_container.primitives

    # Calculate expected primitives
    expected_p = 2.0 ** gamma  # p = rhoY ** gamma = 2.0 ** 1.4 ≈ 2.639
    expected_X_over_rho = 3.0 / 2.0  # 1.5
    expected_Y_over_rho = 2.0 / 2.0  # 1.0
    expected_u = 4.0 / 2.0  # 2.0
    expected_v = 6.0 / 2.0  # 3.0
    expected_w = 8.0 / 2.0  # 4.0

    # Assertions for 3D
    assert np.allclose(prim[..., 0], expected_p, atol=1e-6)
    assert np.allclose(prim[..., 1], expected_X_over_rho, atol=1e-6)
    assert np.allclose(prim[..., 2], expected_Y_over_rho, atol=1e-6)
    assert np.allclose(prim[..., 3], expected_u, atol=1e-6)
    assert np.allclose(prim[..., 4], expected_v, atol=1e-6)
    assert np.allclose(prim[..., 5], expected_w, atol=1e-6)



if __name__ == '__main__':
    test_variables_to_primitive(3, 6, 3, 1.4)
