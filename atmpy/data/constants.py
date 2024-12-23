from enum import IntEnum


class VariableIndices(IntEnum):
    """Create constants for variable indices in the variable container (ndarray) for better readability
    """

    # Indices of Conserved Quantities
    RHO = 0                 # Density
    RHOX = 1                # Potential Temperature inversed? or Exner Pressure?
    RHOY = 2                # Potential Temperature
    RHOU = 3                # Horizontal Velocity x-component
    RHOV = 4                # Horizontal Velocity y-component
    RHOW = 5                # Vertical Velocity

    @classmethod
    def values(cls):
        return [color.value for color in cls]


