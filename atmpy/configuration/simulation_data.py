from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np
from atmpy.infrastructure.enums import BoundaryConditions as BdryType, BoundarySide
from atmpy.infrastructure.enums import SlopeLimiters as LimiterType


@dataclass
class SpatialGrid:
    inx: int = 65
    iny: int = 65
    inz: int = 1
    xmin: float = -1.0
    xmax: float = 1.0
    ymin: float = 0.0
    ymax: float = 1.0
    zmin: float = -1.0
    zmax: float = 1.0


class BoundaryConditions:
    conditions: Dict[BoundarySide, BdryType] = field(
        default_factory=lambda: {
            BoundarySide.LEFT: BdryType.INFLOW,
            BoundarySide.RIGHT: BdryType.OUTFLOW,
            BoundarySide.TOP: BdryType.SLIP_WALL,
            BoundarySide.BOTTOM: BdryType.SLIP_WALL,
            # Add FRONT and BACK if needed
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
