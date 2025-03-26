from enum import IntEnum, Enum


class VariableIndices(IntEnum):
    """Create constants for variable indices in the variable container (ndarray) for better readability"""

    # Indices of Conserved Quantities
    RHO = 0  # Density
    RHOX = 1  # Extra variables for future
    RHOY = 2  # Potential Temperature
    RHOU = 3  # Horizontal Velocity x-component
    RHOV = 4  # Horizontal Velocity y-component
    RHOW = 5  # Vertical Velocity

# For now the assumption is that the number of cell variables and the number of primitive variables are the same
class PrimitiveVariableIndices(IntEnum):
    """Create constants for variable indices in the variable container (ndarray) for better readability"""

    # Indices of Conserved Quantities
    P = 0  # Real Pressure
    X = 1  # Any Extra variables
    Y = 2  # Potential Temperature
    U = 3  # Horizontal Velocity x-component
    V = 4  # Horizontal Velocity y-component
    W = 5  # Vertical Velocity


class HydrostateIndices(IntEnum):
    """Create constants for variable indices in the variable container (ndarray) for better readability"""

    # Indices of Multiple Pressure Variables
    P0 = 0
    P2_0 = 1
    RHO0 = 2
    RHOY0 = 3
    Y0 = 4
    S0 = 5
    S1_0 = 6


class SlopeLimiters(Enum):
    """Map name to function for the slope limiter"""

    MINMOD = "minmod"
    VAN_LEER = "van_leer"
    SUPERBEE = "superbee"
    MC_LIMITER = "mc_limiter"


class RiemannSolvers(Enum):
    """Map name to function for the Riemann solver"""

    RUSANOV = "rusanov"
    MODIFIED_HLL = "modified_hll"
    HLL = "hll"
    HLLC = "hllc"
    ROE = "roe"


class FluxReconstructions(Enum):
    """Map name to function for the reconstruction"""

    MODIFIED_MUSCL = "modified_muscle"
    PIECEWISE_CONSTANT = "piecewise_constant"
    MUSCL = "muscl"


class BoundaryConditions(Enum):
    """Map name to class for boundary conditions."""

    ABSTRACT = "abstract"
    WALL = "wall"
    REYLEICH = "reyleich"
    INFLOW = "inflow"
    OUTFLOW = "outflow"
    NON_REFLECTIVE_OUTLET = "non_reflective_outlet"
    PERIODIC = "periodic"
    REFLECTIVE_GRAVITY = "gravity_reflective_boundary"


class BoundarySide(Enum):
    """Map name to boundary direction."""

    LEFT = "left"  # Beginning of array in x-axis
    RIGHT = "right"  # End of the array in x-axis
    BOTTOM = "bottom"  # Beginning of the array in y-axis
    TOP = "top"  # End of the array in y-axis
    FRONT = "front"  # Beginning of the array in z-axis
    BACK = "back"  # End of the array in z-axis

    @property
    def opposite(self):
        opposites = {
            BoundarySide.LEFT: BoundarySide.RIGHT,
            BoundarySide.RIGHT: BoundarySide.LEFT,
            BoundarySide.BOTTOM: BoundarySide.TOP,
            BoundarySide.TOP: BoundarySide.BOTTOM,
            BoundarySide.FRONT: BoundarySide.BACK,
            BoundarySide.BACK: BoundarySide.FRONT,
        }
        return opposites[self]


class AdvectionRoutines(Enum):
    """Map name to function for the advection solver"""

    STRANG_SPLIT = "strang_split"
