from enum import IntEnum, Enum


class VariableIndices1D(Enum):
    """Create constants for 1D variable indices in the variable container (ndarray) for better readability

    Notes
    -----
    We need to separate the index container for different dimensions since the variable container holds the velocities
    one after the other. So in 1D, index i = 2 is P since no other velocity exist, in 2d and 3d is RHOV. Similar thing
    happens for RHOW and index 3. The indices of P and PX are shifted by the dimension.
    """

    RHO = 0
    RHOU = 1
    P = 2
    PX = 3

    @classmethod
    def values(cls):
        return [color.value for color in cls]


class VariableIndices2D(Enum):
    """Create constants for 2D variable indices in the variable container (ndarray) for better readability

    Notes
    -----
    We need to separate the index container for different dimensions since the variable container holds the velocities
    one after the other. So in 1D, index i = 2 is P since no other velocity exist, in 2d and 3d is RHOV. Similar thing
    happens for RHOW and index 3. The indices of P and PX are shifted by the dimension.
    """

    RHO = 0
    RHOU = 1
    RHOV = 2
    P = 3
    PX = 4

    @classmethod
    def values(cls):
        return [color.value for color in cls]


class VariableIndices3D(Enum):
    """Create constants for 3D variable indices in the variable container (ndarray) for better readability

    Notes
    -----
    We need to separate the index container for different dimensions since the variable container holds the velocities
    one after the other. So in 1D, index i = 2 is P since no other velocity exist, in 2d and 3d is RHOV. Similar thing
    happens for RHOW and index 3. The indices of P and PX are shifted by the dimension.
    """

    RHO = 0
    RHOU = 1
    RHOV = 2
    RHOW = 3
    P = 4
    PX = 5

    @classmethod
    def values(cls):
        return [color.value for color in cls]
