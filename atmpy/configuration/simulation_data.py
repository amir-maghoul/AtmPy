"""The module containing the simulation data. The simulation data are the numerical, physical and temporal data needed
for a concrete simulation."""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np
from atmpy.infrastructure.enums import (
    BoundaryConditions as BdryType,
    BoundarySide,
    AdvectionRoutines,
)
from atmpy.infrastructure.enums import SlopeLimiters as LimiterType
from atmpy.grid.utility import DimensionSpec, create_grid
from atmpy.grid.kgrid import Grid
from atmpy.physics.gravity import Gravity
from atmpy.physics.thermodynamics import Thermodynamics
from atmpy.time_integrators.coriolis import CoriolisOperator


@dataclass
class SpatialGrid:
    """The data class for spatial grid information."""

    ndim: int = 2
    nx: int = 64
    ny: int = 10
    nz: int = 0  # Keep 0 for 2D

    # Number of ghost cells on each side
    ngx: int = 2
    ngy: int = 2
    ngz: int = 0  # Keep 0 for 2D

    # Ranges: x direction
    xmin: float = -0.5  # Updated default
    xmax: float = 0.5  # Updated default

    # Ranges: y direction
    ymin: float = -0.5  # Updated default
    ymax: float = 0.5  # Updated default

    # Ranges: z direction
    zmin: float = -0.5  # Keep default
    zmax: float = 0.5  # Keep default

    # Grid object created after initialization
    grid: Grid = field(init=False)

    def __post_init__(self):
        dims = self._create_dimension_specs()
        self.grid = create_grid(dims)

    def _create_dimension_specs(self):
        """Create dimensional specifications for the grid."""
        dims = []
        if self.nx > 0:
            dims.append(DimensionSpec(self.nx, self.xmin, self.xmax, self.ngx))
        if self.ny > 0 and self.ndim >= 2:
            dims.append(DimensionSpec(self.ny, self.ymin, self.ymax, self.ngy))
        if self.nz > 0 and self.ndim == 3:
            dims.append(DimensionSpec(self.nz, self.zmin, self.zmax, self.ngz))

        if len(dims) != self.ndim:
            # Adjust ndim if nx/ny/nz settings imply a different dimension
            self.ndim = len(dims)
            if self.ndim == 0:
                raise ValueError("Grid must have at least one dimension (nx > 0).")
        return dims


@dataclass
class BoundarySpec:
    """Specifies boundary conditions for a single side."""

    main_type: BdryType
    mpv_type: Optional[BdryType] = (
        None  # Explicitly None if same as main or default desired
    )


@dataclass
class BoundaryConditions:
    """Stores the boundary condition specifications for each side."""

    # Dictionary mapping BoundarySide enum to BoundarySpec dataclass
    conditions: Dict[BoundarySide, BoundarySpec] = field(
        default_factory=lambda: {
            # Default to PERIODIC for both main and mpv, matching vortex case
            BoundarySide.LEFT: BoundarySpec(
                main_type=BdryType.PERIODIC, mpv_type=BdryType.PERIODIC
            ),
            BoundarySide.RIGHT: BoundarySpec(
                main_type=BdryType.PERIODIC, mpv_type=BdryType.PERIODIC
            ),
            BoundarySide.BOTTOM: BoundarySpec(
                main_type=BdryType.PERIODIC, mpv_type=BdryType.PERIODIC
            ),
            BoundarySide.TOP: BoundarySpec(
                main_type=BdryType.PERIODIC, mpv_type=BdryType.PERIODIC
            ),
            # Add FRONT/BACK if needed for 3D
        }
    )


@dataclass
class Temporal:
    """The data class for time information."""

    CFL: float = 0.45
    dtfixed0: float = 0.01
    dtfixed: float = None
    acoustic_timestep: float = 0.0
    tout: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    tmax: float = 0.05
    stepmax: int = 101


@dataclass
class GlobalConstants:
    """The data class for global constants."""

    grav: float = 9.81
    omega: float = 0.0
    R_gas: float = 287.4
    R_vap: float = 461.0
    Q_vap: float = 2.53e6
    gamma: float = 1.4
    cp_gas: float = field(init=False)
    p_ref: float = 1e5
    T_ref: float = 300.0
    h_ref: float = 10000.0
    t_ref: float = 100.0
    rho_ref: float = field(init=False)
    u_ref: float = field(init=False)
    Nsq_ref: float = 0.0
    N_ref: float = field(init=False)
    Cs: float = field(init=False)

    def __post_init__(self):
        # Ensure gamma != 1 before calculating cp_gas
        if np.isclose(self.gamma, 1.0):
            raise ValueError("Gamma (ratio of specific heats) cannot be 1.")
        self.cp_gas = self.gamma * self.R_gas / (self.gamma - 1.0)

        # Check for valid T_ref and R_gas before calculating rho_ref
        if self.T_ref <= 0 or self.R_gas <= 0:
            raise ValueError("T_ref and R_gas must be positive to calculate rho_ref.")
        self.rho_ref = self.p_ref / (self.R_gas * self.T_ref)

        # Check for valid t_ref before calculating u_ref
        if self.t_ref == 0:
            raise ValueError("t_ref cannot be zero.")
        self.u_ref = self.h_ref / self.t_ref

        # Check for valid cp_gas and T_ref before calculating N_ref
        if self.cp_gas <= 0 or self.T_ref <= 0:
            raise ValueError("cp_gas and T_ref must be positive to calculate N_ref.")
        # Avoid sqrt of negative if grav is negative, use absolute value
        self.Nsq_ref = (
            abs(self.grav * self.grav / (self.cp_gas * self.T_ref))
            if self.grav != 0
            else 0.0
        )
        self.N_ref = np.sqrt(self.Nsq_ref)

        # Check for valid gamma, R_gas, T_ref before calculating Cs
        if self.gamma <= 0 or self.R_gas <= 0 or self.T_ref <= 0:
            raise ValueError(
                "gamma, R_gas, and T_ref must be positive to calculate Cs."
            )
        self.Cs = np.sqrt(self.gamma * self.R_gas * self.T_ref)


@dataclass
class ModelRegimes:
    """The data class for model regimes information."""

    is_ArakawaKonor: int = 0
    is_nongeostrophic: int = 1
    is_nonhydrostatic: int = 1
    is_compressible: int = 1
    Msq: float = 0.115

    def update_derived_fields(self, constants: GlobalConstants):
        glc = constants
        self.Msq = glc.u_ref * glc.u_ref / (glc.R_gas * glc.T_ref)


@dataclass
class Physics:
    """The data class for physics information."""

    wind_speed: Union[List[float], np.ndarray, Tuple[float]] = (0.0, 0.0, 0.0)
    gravity_strength: Union[List[float], np.ndarray, Tuple[float]] = (0.0, 0.0, 0.0)
    coriolis_strength: Union[List[float], np.ndarray, Tuple[float]] = (0.0, 0.0, 1.0)
    stratification: callable = lambda y: 1.0
    gravity: Gravity = field(init=False)
    coriolis: CoriolisOperator = field(init=False)

    def update_derived_fields(self, constants: GlobalConstants, grid_cfg: SpatialGrid):
        """Recalculates gravity and coriolis based on current config."""
        g = constants.grav * constants.h_ref / (constants.R_gas * constants.T_ref)
        ndim = grid_cfg.ndim
        self.gravity = Gravity(self.gravity_strength, ndim)
        self.gravity.strength = g
        self.gravity.vector[self.gravity.axis] = g
        self.gravity_strength = self.gravity.vector
        self.coriolis = CoriolisOperator(self.coriolis_strength, self.gravity)
        print("Physics derived fields updated.")


@dataclass
class Numerics:
    """The data class for numerical information."""

    do_advection: bool = True
    limiter_scalars: LimiterType = LimiterType.VAN_LEER
    first_order_advection_routine: AdvectionRoutines = AdvectionRoutines.FIRST_ORDER_RK
    second_order_advection_routine: AdvectionRoutines = AdvectionRoutines.STRANG_SPLIT
    tol: float = 1e-8
    max_iterations: int = 6000
    initial_projection: bool = True
    num_vars_cell: int = 6


@dataclass
class Diagnostics:
    """The data class for diagnostic information."""

    diag: bool = True
    diag_plot_compare: bool = False
    diag_current_run: str = "atmpy_run"
    analysis: bool = False


@dataclass
class Outputs:
    """The data class for output information."""

    autogen_fn: bool = False
    output_timesteps: bool = True
    output_type: str = "test"
    output_path: str = "/home/amir/Projects/Python/Atmpy/atmpy/output_data/"
    output_folder: str = ""
    output_base_name: str = "_travelling_vortex"
    output_filename: str = ""
    output_suffix: str = ""
    output_extension: str = ".nc"
    output_frequency_steps: int = 100
    checkpoint_base_name: str = "_traveling_vortex_checkpoint"
    enable_checkpointing: str = True
    checkpoint_frequency_steps: int = 100
    checkpoint_filename: str = ""
    aux: str = ""
