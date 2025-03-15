"""The module containing the simulation data. The simulation data are the numerical, physical and temporal data needed
for a concrete simulation."""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any
import numpy as np
from atmpy.infrastructure.enums import BoundaryConditions as BdryType, BoundarySide
from atmpy.infrastructure.enums import SlopeLimiters as LimiterType
from atmpy.grid.utility import DimensionSpec, create_grid


@dataclass
class BoundaryFace:
    face_id: int
    normal_vector: Tuple[float, float, float]


@dataclass
class BoundaryConditionStructure:
    type: BdryType
    params: Dict[str, Any] = field(default_factory=dict)
    faces: List[BoundaryFace] = field(default_factory=list)


@dataclass
class SpatialGrid:
    ndim = 2
    # Number of points
    nx: int = 65
    ny: int = 65
    nz: int = 0

    # Number of ghost cells
    ngx = 2
    ngy = 2
    ngz = 0

    # Ranges: x direction
    xmin: float = -1.0
    xmax: float = 1.0

    # Ranges: y direction
    ymin: float = 0.0
    ymax: float = 1.0

    # Ranges: z direction
    zmin: float = 0.0
    zmax: float = 0.0

    def __post_init__(self):
        dims = self.initialize()
        self.grid = create_grid(dims)

    def initialize(self):
        """Create dimensional specifications in form of DimensionSpec objects to create a grid object."""
        dim_x = DimensionSpec(self.nx, self.xmin, self.xmax, self.ngx)
        dim_y = DimensionSpec(self.ny, self.ymin, self.ymax, self.ngy)
        dim_z = DimensionSpec(self.nz, self.zmin, self.zmax, self.ngz)
        dims = [dim_x, dim_y, dim_z]
        return dims[: self.ndim]


@dataclass
class BoundaryConditions:
    conditions: Dict[BoundarySide, BdryType] = field(
        default_factory=lambda: {
            BoundarySide.LEFT: BoundaryConditionStructure(type=BdryType.INFLOW),
            BoundarySide.RIGHT: BoundaryConditionStructure(type=BdryType.INFLOW),
            BoundarySide.TOP: BoundaryConditionStructure(type=BdryType.INFLOW),
            BoundarySide.BOTTOM: BoundaryConditionStructure(type=BdryType.INFLOW),
            # Add FRONT and BACK if needed
        }
    )

@dataclass
class BoundaryConditionsData:
# Here we use a dict with keys from BoundarySide.
    conditions: Dict[Any, BoundaryConditionStructure] = field(
        default_factory=lambda: {
            'LEFT': BoundaryConditionStructure(type="INFLOW", params={"inflow_value": 1.0}),
            'RIGHT': BoundaryConditionStructure(type="OUTFLOW", params={"outflow_value": 0.0}),
            'TOP': BoundaryConditionStructure(type="SLIP_WALL", params={"wall_coeff": 0.5}),
            'BOTTOM':BoundaryConditionStructure(type="SLIP_WALL", params={"wall_coeff": 0.5}),
            'FRONT': BoundaryConditionStructure(type="PERIODIC", params={}),
            'BACK': BoundaryConditionStructure(type="PERIODIC", params={}),
        }
    )


@dataclass
class Temporal:
    CFL: float = 0.5
    dtfixed0: float = 100.0
    dtfixed: float = 100.0
    acoustic_timestep: float = 0.0
    tout: np.ndarray = field(default_factory=lambda: np.arange(0.0, 1.01, 0.01)[10:])
    stepmax: int = 10000


@dataclass
class ModelRegimes:
    is_ArakawaKonor: int = 0
    is_nonhydrostatic: int = 1
    is_compressible: int = 1
    compressibility: float = 0.0


@dataclass
class Physics:
    u_wind_speed: float = 0.0
    v_wind_speed: float = 0.0
    w_wind_speed: float = 0.0
    stratification: callable = lambda y: 1.0  # Placeholder for the function


@dataclass
class Numerics:
    do_advection: bool = True
    limiter: LimiterType = LimiterType.VAN_LEER
    tol: float = 1e-8
    max_iterations: int = 6000


@dataclass
class Diagnostics:
    diag: bool = False
    diag_plot_compare: bool = False


@dataclass
class Outputs:
    autogen_fn: bool = False
    output_timesteps: bool = False
    output_type: str = "output"
    output_suffix: str = "_64_64"  # Example based on inx and iny


@dataclass
class GlobalConstants:
    # Define global constants here
    grav: float = 9.81
    omega: float = 7.292e-5
    R_gas: float = 287.4
    R_vap: float = 461.0
    Q_vap: float = 2.53e6
    gamma: float = 1.4
    cp_gas: float = field(init=False)
    p_ref: float = 1e5
    T_ref: float = 300.0
    h_ref: float = 10.0e3
    t_ref: float = 100.0

    def __post_init__(self):
        self.cp_gas = self.gamma * self.R_gas / (self.gamma - 1.0)
        self.rho_ref = self.p_ref / (self.R_gas * self.T_ref)
        self.N_ref = self.grav / np.sqrt(self.cp_gas * self.T_ref)
        self.Cs = np.sqrt(self.gamma * self.R_gas * self.T_ref)
