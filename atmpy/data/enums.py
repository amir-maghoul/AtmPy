from enum import IntEnum, Enum


class VariableIndices(IntEnum):
    """Create constants for variable indices in the variable container (ndarray) for better readability"""

    # Indices of Conserved Quantities
    RHO = 0  # Density
    RHOX = 1  # Potential Temperature inversed? or Exner Pressure?
    RHOY = 2  # Potential Temperature
    RHOU = 3  # Horizontal Velocity x-component
    RHOV = 4  # Horizontal Velocity y-component
    RHOW = 5  # Vertical Velocity


class PrimitiveVariableIndices(IntEnum):
    """Create constants for variable indices in the variable container (ndarray) for better readability"""

    # Indices of Conserved Quantities
    P = 0  # Density
    X = 1  # Potential Temperature inversed? or Exner Pressure?
    Y = 2  # Potential Temperature
    U = 3  # Horizontal Velocity x-component
    V = 4  # Horizontal Velocity y-component
    W = 5  # Vertical Velocity


class SlopeLimiters(Enum):
    """Map name to function for the slope limiter"""

    MINMOD = "minmod"
    VAN_LEER = "van_leer"
    SUPERBEE = "superbee"
    MC_LIMITER = "mc_limiter"


class RiemannSolvers(Enum):
    """Map name to function for the Riemann solver"""

    RUSANOV = "rusanov"
    HLL = "hll"
    HLLC = "hllc"
    ROE = "roe"


class FluxReconstructions(Enum):
    """Map name to function for the reconstruction"""

    PIECEWISE_CONSTANT = "piecewise_constant"
    MUSCL = "muscl"
