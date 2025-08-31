"""The module for thermodynamic values"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class Thermodynamics:
    gamma: float = 1.4  # Dry isentropic exponent. Default value 1.4

    # Initialize derived values as private, to be set via update()
    gamminv: float = field(init=False)  # gamma inversed.
    gm1: float = field(init=False)  # gamma minus one. needed for some calculations
    gm1inv: float = field(init=False)  # gamma minus one inversed.
    Gamma: float = field(
        init=False
    )  # Capital gamma. Main exponent of pressure in calculation of Exner pressure
    Gammainv: float = field(init=False)  # Capital gamma inversed.

    def __post_init__(self):
        # Automatically calculate dependent values after initialization
        self.update(self.gamma)

    def update(self, gamma: float):
        """Update gamma and all dependent properties."""
        self.gamma = gamma
        self.gamminv = 1.0 / gamma
        self.gm1 = gamma - 1.0
        self.gm1inv = 1.0 / (gamma - 1.0)
        self.Gamma = (gamma - 1.0) / gamma
        self.Gammainv = gamma / (gamma - 1.0)
