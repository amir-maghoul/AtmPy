""" This is a registry container to map names to their corresponding functions. """

from atmpy.flux.riemann_solvers import *
from atmpy.flux.reconstruction import *
from atmpy.data.enums import SlopeLimiters, RiemannSolvers, FluxReconstructions

SLOPE_LIMITERS = {
    SlopeLimiters.MINMOD: minmod,
    SlopeLimiters.VAN_LEER: van_leer,
    SlopeLimiters.SUPERBEE: superbee,
    SlopeLimiters.MC_LIMITER: mc_limiter,
}

RIEMANN_SOLVERS = {
    RiemannSolvers.RUSANOV: rusanov,
    RiemannSolvers.HLL: hll,
    RiemannSolvers.HLLC: hllc,
    RiemannSolvers.ROE: roe,
}

FLUX_RECONSTRUCTION = {
    FluxReconstructions.PIECEWISE_CONSTANT: piecewise_constant,
    FluxReconstructions.MUSCL: muscl,
}
