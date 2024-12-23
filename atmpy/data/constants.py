from enum import IntEnum


class VariableIndices(IntEnum):
    """Create constants for variable indices in the variable container (ndarray) for better readability
    """

    # Indices of Conserved Quantities
    RHO = 0                 # Density
    RHOX = 2                # Potential Temperature inversed? or Exner Pressure?
    RHOY = 3                # Potential Temperature
    RHOU = 4                # Horizontal Velocity x-component
    RHOV = 5                # Horizontal Velocity y-component
    RHOW = 6                # Vertical Velocity

    @classmethod
    def values(cls):
        return [color.value for color in cls]


