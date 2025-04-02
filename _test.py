import numpy as np
import scipy as sp
import numba as nb
from enum import Enum


class BdryType(Enum):
    """
    An enumeration class that defines the accepted boundary condition types.
    """

    WALL = "symmetric"
    PERIODIC = "wrap"
    RAYLEIGH = "radiation"


def stencil_9pt(grid, mpv, ud, diag_inv, coriolis_params):
    igx = grid.igx
    igy = grid.igy

    icxn = grid.inx
    icyn = grid.iny

    iicxn = icxn - (2 * igx)
    iicyn = icyn - (2 * igy)

    iicxn, iicyn = iicyn, iicxn

    dx = grid.dy
    dy = grid.dx

    inner_domain = (slice(igx, -igx), slice(igy, -igy))
    i1 = node.i1

    hplusx = mpv.wplus[1][i1].reshape(
        -1,
    )
    hplusy = mpv.wplus[0][i1].reshape(
        -1,
    )
    hcenter = mpv.wcenter[i1].reshape(
        -1,
    )

    diag_inv = diag_inv[i1].reshape(
        -1,
    )

    oodx = 1.0 / (dx)
    oody = 1.0 / (dy)

    x_periodic = ud.bdry_type[1] == BdryType.PERIODIC
    y_periodic = ud.bdry_type[0] == BdryType.PERIODIC

    x_wall = ud.bdry_type[1] == BdryType.WALL or ud.bdry_type[1] == BdryType.RAYLEIGH
    y_wall = ud.bdry_type[0] == BdryType.WALL or ud.bdry_type[0] == BdryType.RAYLEIGH

    return lambda p: lap2D_gather(
        p,
        igx,
        igy,
        iicxn,
        iicyn,
        hplusx,
        hplusy,
        hcenter,
        oodx,
        oody,
        x_periodic,
        y_periodic,
        x_wall,
        y_wall,
        diag_inv,
        coriolis_params,
    )


@nb.jit(nopython=True, nogil=False, cache=True)
def lap2D_gather(
    p,
    igx,
    igy,
    iicxn,
    iicyn,
    hplusx,
    hplusy,
    hcenter,
    oodx,
    oody,
    x_periodic,
    y_periodic,
    x_wall,
    y_wall,
    diag_inv,
    coriolis,
):
    ngnc = (iicxn) * (iicyn)
    lap = np.zeros((ngnc))
    cnt_x = 0
    cnt_y = 0

    nine_pt = 0.25 * (2.0) * 1.0
    cyy, cxx, cyx, cxy = coriolis
    oodx2 = 0.5 * oodx**2
    oody2 = 0.5 * oody**2

    for idx in range(iicxn * iicyn):
        ne_topleft = idx - iicxn - 1
        ne_topright = idx - iicxn
        ne_botleft = idx - 1
        ne_botright = idx

        # get indices of the 9pt stencil
        topleft_idx = idx - iicxn - 1
        midleft_idx = idx - 1
        botleft_idx = idx + iicxn - 1

        topmid_idx = idx - iicxn
        midmid_idx = idx
        botmid_idx = idx + iicxn

        topright_idx = idx - iicxn + 1
        midright_idx = idx + 1
        botright_idx = idx + iicxn + 1

        if cnt_x == 0:
            topleft_idx += iicxn - 1
            midleft_idx += iicxn - 1
            botleft_idx += iicxn - 1

            ne_topleft += iicxn - 1
            ne_botleft += iicxn - 1

        if cnt_x == (iicxn - 1):
            topright_idx -= iicxn - 1
            midright_idx -= iicxn - 1
            botright_idx -= iicxn - 1

            ne_topright -= iicxn - 1
            ne_botright -= iicxn - 1

        if cnt_y == 0:
            topleft_idx += (iicxn) * (iicyn - 1)
            topmid_idx += (iicxn) * (iicyn - 1)
            topright_idx += (iicxn) * (iicyn - 1)

            ne_topleft += (iicxn) * (iicyn - 1)
            ne_topright += (iicxn) * (iicyn - 1)

        if cnt_y == (iicyn - 1):
            botleft_idx -= (iicxn) * (iicyn - 1)
            botmid_idx -= (iicxn) * (iicyn - 1)
            botright_idx -= (iicxn) * (iicyn - 1)

            ne_botleft -= (iicxn) * (iicyn - 1)
            ne_botright -= (iicxn) * (iicyn - 1)

        topleft = p[topleft_idx]
        midleft = p[midleft_idx]
        botleft = p[botleft_idx]

        topmid = p[topmid_idx]
        midmid = p[midmid_idx]
        botmid = p[botmid_idx]

        topright = p[topright_idx]
        midright = p[midright_idx]
        botright = p[botright_idx]

        hplusx_topleft = hplusx[ne_topleft]
        hplusx_botleft = hplusx[ne_botleft]
        hplusy_topleft = hplusy[ne_topleft]
        hplusy_botleft = hplusy[ne_botleft]

        hplusx_topright = hplusx[ne_topright]
        hplusx_botright = hplusx[ne_botright]
        hplusy_topright = hplusy[ne_topright]
        hplusy_botright = hplusy[ne_botright]

        cxx_tl = cxx[ne_topleft]
        cxx_tr = cxx[ne_topright]
        cxx_bl = cxx[ne_botleft]
        cxx_br = cxx[ne_botright]

        cxy_tl = cxy[ne_topleft]
        cxy_tr = cxy[ne_topright]
        cxy_bl = cxy[ne_botleft]
        cxy_br = cxy[ne_botright]

        cyx_tl = cyx[ne_topleft]
        cyx_tr = cyx[ne_topright]
        cyx_bl = cyx[ne_botleft]
        cyx_br = cyx[ne_botright]

        cyy_tl = cyy[ne_topleft]
        cyy_tr = cyy[ne_topright]
        cyy_bl = cyy[ne_botleft]
        cyy_br = cyy[ne_botright]

        if x_wall and (cnt_x == 0):
            hplusx_topleft = 0.0
            hplusy_topleft = 0.0
            hplusx_botleft = 0.0
            hplusy_botleft = 0.0

        if x_wall and (cnt_x == (iicxn - 1)):
            hplusx_topright = 0.0
            hplusy_topright = 0.0
            hplusx_botright = 0.0
            hplusy_botright = 0.0

        if y_wall and (cnt_y == 0):
            hplusx_topleft = 0.0
            hplusy_topleft = 0.0
            hplusx_topright = 0.0
            hplusy_topright = 0.0

        if y_wall and (cnt_y == (iicyn - 1)):
            hplusx_botleft = 0.0
            hplusy_botleft = 0.0
            hplusx_botright = 0.0
            hplusy_botright = 0.0

        Dx_tl = 0.5 * (topmid - topleft + midmid - midleft) * hplusx_topleft
        Dx_tr = 0.5 * (topright - topmid + midright - midmid) * hplusx_topright
        Dx_bl = 0.5 * (botmid - botleft + midmid - midleft) * hplusx_botleft
        Dx_br = 0.5 * (botright - botmid + midright - midmid) * hplusx_botright

        Dy_tl = 0.5 * (midmid - topmid + midleft - topleft) * hplusy_topleft
        Dy_tr = 0.5 * (midright - topright + midmid - topmid) * hplusy_topright
        Dy_bl = 0.5 * (botmid - midmid + botleft - midleft) * hplusy_botleft
        Dy_br = 0.5 * (botright - midright + botmid - midmid) * hplusy_botright

        fac = 1.0
        Dxx = (
            0.5
            * (cxx_tr * Dx_tr - cxx_tl * Dx_tl + cxx_br * Dx_br - cxx_bl * Dx_bl)
            * oodx
            * oodx
            * fac
        )
        Dyy = (
            0.5
            * (cyy_br * Dy_br - cyy_tr * Dy_tr + cyy_bl * Dy_bl - cyy_tl * Dy_tl)
            * oody
            * oody
            * fac
        )
        Dyx = (
            0.5
            * (cyx_br * Dy_br - cyx_bl * Dy_bl + cyx_tr * Dy_tr - cyx_tl * Dy_tl)
            * oody
            * oodx
            * fac
        )
        Dxy = (
            0.5
            * (cxy_br * Dx_br - cxy_tr * Dx_tr + cxy_bl * Dy_bl - cxy_tl * Dx_tl)
            * oodx
            * oody
            * fac
        )

        lap[idx] = Dxx + Dyy + Dyx + Dxy + hcenter[idx] * p[idx]
        lap[idx] *= diag_inv[idx]

        cnt_x += 1
        if cnt_x % iicxn == 0:
            cnt_y += 1
            cnt_x = 0

    return lap


if __name__ == "__main__":
    from atmpy.physics.thermodynamics import Thermodynamics
    from atmpy.grid.utility import DimensionSpec, create_grid
    from atmpy.variables.variables import Variables
    from atmpy.infrastructure.enums import VariableIndices as VI
    from atmpy.boundary_conditions.utility import create_params
    from atmpy.variables.multiple_pressure_variables import MPV

    np.set_printoptions(linewidth=100)

    nx = 1
    ngx = 2
    nnx = nx + 2 * ngx
    ny = 2
    ngy = 2
    nny = ny + 2 * ngy

    dim = [DimensionSpec(nx, 0, 2, ngx), DimensionSpec(ny, 0, 2, ngy)]
    grid = create_grid(dim)
    rng = np.random.default_rng()
    arr = np.arange(nnx * nny)
    rng.shuffle(arr)
    array = arr.reshape(nnx, nny)

    variables = Variables(grid, 6, 1)
    variables.cell_vars[..., VI.RHO] = 1
    variables.cell_vars[..., VI.RHO][1:-1, 1:-1] = 4
    variables.cell_vars[..., VI.RHOU] = array
    variables.cell_vars[..., VI.RHOY] = 2
    variables.cell_vars[..., VI.RHOW] = 3

    rng.shuffle(arr)
    array = arr.reshape(nnx, nny)
    variables.cell_vars[..., VI.RHOV] = array
    gravity = np.array([0.0, 1.0, 0.0])
    th = Thermodynamics()

    stratification = lambda x: x**2
    mpv = MPV(grid)
