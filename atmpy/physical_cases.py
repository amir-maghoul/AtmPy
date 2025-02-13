import numpy as np
import matplotlib.pyplot as plt
from atmpy.grid.utility import DimensionSpec, create_grid
from atmpy.physics.eos import (
    ExnerBasedEOS,
)  # Ensure this EOS is suitable for Sod's problem
from atmpy.variables.variables import Variables
from atmpy.infrastructure.enums import (
    VariableIndices as VI,
    SlopeLimiters,
    RiemannSolvers,
    FluxReconstructions,
)
from atmpy.flux.flux import Flux  # Replace with the correct import path if different
from atmpy.grid.kgrid import Grid
from atmpy.physics.eos import EOS


def steady_state():
    dim = [DimensionSpec(1, 0, 2, 2), DimensionSpec(2, 0, 2, 2)]
    grid = create_grid(dim)

    dt = 0.01

    variables = Variables(grid, 5, 1)
    variables.cell_vars[..., VI.RHO] = 1
    variables.cell_vars[..., VI.RHOU] = 2
    variables.cell_vars[..., VI.RHOY] = 3

    eos = ExnerBasedEOS()
    flux = Flux(grid, variables, eos, dt)
    variables.to_primitive(eos)
    primitives = variables.primitives

    flux.apply_riemann_solver(1, "x")
    print(
        flux.flux["x"][..., VI.RHO]
    )  # the variable rho of the flux container is rho*u.
    print(flux.flux["x"][..., VI.RHOY])  # flux.rhoY = rhoY*u = 6
    print(flux.flux["y"][..., VI.RHOU])


def main():
    steady_state()


if __name__ == "__main__":
    main()
