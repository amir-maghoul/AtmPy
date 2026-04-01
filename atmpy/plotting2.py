import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

# ==========================================
# 1. USER CONFIGURATION
# ==========================================
# List your files and their corresponding titles here
SIMULATIONS = [
    {
        "path": "/home/amir/Projects/Python/Atmpy/atmpy/output_data/traveling_vortex_3d/_traveling_vortex_3d_40_3_40_cor_1_full_comp.nc",
        "title": "Full Compressible"
    },

    {
        "path": "/home/amir/Projects/Python/Atmpy/atmpy/output_data/traveling_vortex_3d/_traveling_vortex_3d_40_3_40_cor_1_incomp.nc",
        "title": "Pseudo Incompressible"
    },
    {
        "path": "/home/amir/Projects/Python/Atmpy/atmpy/output_data/traveling_vortex_3d/_traveling_vortex_3d_40_3_40_cor_1_qg.nc",
        "title": "Quasi-Geostrophic"
    }
]

# --- NEW FLAGS ---
SINGLE_VAR_MODE = True
SINGLE_VAR_TARGET = 'p2_nodes'
# -----------------

# Visuals
USE_COLORBLIND_SAFE = True

# Plotting Selection
TIME_INDEX = -1
SLICE_DIM = 'y'
SLICE_VAL = 0.0

# Style Constants
UNIT_LABEL = "($10$ km)"
N_LEVELS = 25
LINE_WIDTH = 0.2
LINE_COLOR = 'k'

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 14,
    'axes.titlesize': 16, 'axes.labelsize': 14,
    'xtick.labelsize': 12, 'ytick.labelsize': 12,
    'figure.dpi': 300
})

# ==========================================
# 2. VARIABLE CONFIGURATION
# ==========================================
if USE_COLORBLIND_SAFE:
    CMAP_SEQ = 'cividis'
    CMAP_DIV = 'cividis'  # Standard choice for divergence
else:
    CMAP_SEQ = 'viridis'
    CMAP_DIV = 'RdBu_r'

# Columns for the standard 2x3 comparison
VARS_CONFIG = [
    {'var': 'p2_nodes', 'title': r"$\pi'$ (Exner)", 'type': 'diverging', 'pert': False},
    {'var': 'rho', 'title': r"$\rho'$ (Density)", 'type': 'diverging', 'pert': True},
    {'var': 'u', 'title': r"$u$ (Velocity)", 'type': 'diverging', 'pert': False}
]


# ==========================================
# 3. DATA PROCESSING HELPERS
# ==========================================
def get_plot_label(sim_dim_name):
    clean_name = sim_dim_name.replace('_nodes', '').replace('_node', '').lower()
    mapping = {'y': 'z', 'z': 'y', 'x': 'x'}
    for k, v in mapping.items():
        if k in clean_name: return v
    return clean_name


def calculate_perturbation(da):
    horiz_dims = [d for d in da.dims if 'x' in d.lower() or 'z' in d.lower()]
    return da - da.mean(dim=horiz_dims) if horiz_dims else da


def get_slice_data(ds, var_name, t_idx, slice_dim, slice_val, calc_perturbation=False):
    if var_name not in ds: return None
    da = ds[var_name]
    if 'time' in da.dims:
        t_idx = da.sizes['time'] + t_idx if t_idx < 0 else t_idx
        da = da.isel(time=t_idx)
    if calc_perturbation:
        da = calculate_perturbation(da)

    candidates = [d for d in da.dims if slice_dim in d]
    if not candidates:
        da_slice = da
    else:
        da_slice = da.sel({candidates[0]: slice_val}, method='nearest')

    rem_dims = list(da_slice.dims)
    if len(rem_dims) != 2: return None

    sim_dim_v = next((d for d in rem_dims if 'y' in d.lower()), None)
    if sim_dim_v:
        dim_vert = sim_dim_v
        dim_horiz = [d for d in rem_dims if d != dim_vert][0]
    else:
        sim_dim_z = next((d for d in rem_dims if 'z' in d.lower()), None)
        dim_vert = sim_dim_z if sim_dim_z else rem_dims[0]
        dim_horiz = [d for d in rem_dims if d != dim_vert][0]

    X, Y, Z_data = da_slice[dim_horiz].values, da_slice[dim_vert].values, da_slice.values
    if rem_dims.index(dim_horiz) < rem_dims.index(dim_vert):
        Z_data = Z_data.T

    return {'X': X, 'Y': Y, 'data': Z_data,
            'xlabel': f"{get_plot_label(dim_horiz)} {UNIT_LABEL}",
            'ylabel': f"{get_plot_label(dim_vert)} {UNIT_LABEL}"}


# ==========================================
# 4. PLOTTING LOGIC
# ==========================================
def plot_panel(ax, spec, plot_data, is_bottom, is_left):
    if plot_data is None:
        ax.text(0.5, 0.5, "Data Missing", ha='center', va='center')
        ax.set_box_aspect(1);
        return

    X, Y, Z = plot_data['X'], plot_data['Y'], plot_data['data']
    dmin, dmax = np.nanmin(Z), np.nanmax(Z)
    levels = np.linspace(dmin, dmax, N_LEVELS) if not np.isclose(dmin, dmax) else np.linspace(dmin - 0.1, dmax + 0.1,
                                                                                              N_LEVELS)

    cmap_to_use = CMAP_DIV if spec['type'] == 'diverging' else CMAP_SEQ
    cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap_to_use, extend='both')
    if not np.isclose(dmin, dmax):
        ax.contour(X, Y, Z, levels=levels, colors=LINE_COLOR, linewidths=LINE_WIDTH)

    ax.set_box_aspect(1)
    if is_bottom:
        ax.set_xlabel(plot_data['xlabel'])
    else:
        ax.set_xticklabels([])
    if is_left:
        ax.set_ylabel(plot_data['ylabel'])
    else:
        ax.set_yticklabels([])

    cbar = plt.colorbar(cf, ax=ax, orientation='vertical', shrink=0.8, pad=0.04)
    cbar.ax.ticklabel_format(style='sci', scilimits=(-1, 1), axis='y', useMathText=True)


def create_plots():
    # Filter only existing files
    valid_sims = [s for s in SIMULATIONS if Path(s['path']).exists()]
    if not valid_sims:
        print("No valid files found.");
        return

    datasets = [xr.open_dataset(s['path']) for s in valid_sims]

    if SINGLE_VAR_MODE:
        # Beamer Layout: 1 Row, N Columns
        fig, axes = plt.subplots(nrows=1, ncols=len(valid_sims),
                                 figsize=(5 * len(valid_sims), 5), constrained_layout=True)
        if len(valid_sims) == 1: axes = [axes]

        # Identify the config for the target variable
        target_cfg = next((v for v in VARS_CONFIG if v['var'] == SINGLE_VAR_TARGET),
                          {'var': SINGLE_VAR_TARGET, 'title': SINGLE_VAR_TARGET, 'type': 'diverging', 'pert': False})

        for i, (ds, sim_info) in enumerate(zip(datasets, valid_sims)):
            data = get_slice_data(ds, target_cfg['var'], TIME_INDEX, SLICE_DIM, SLICE_VAL,
                                  calc_perturbation=target_cfg['pert'])
            plot_panel(axes[i], target_cfg, data, is_bottom=True, is_left=(i == 0))
            axes[i].set_title(sim_info['title'], fontsize=16, fontweight='bold')

        save_name = f"beamer_{SINGLE_VAR_TARGET}.png"
    else:
        # Standard 2x3 Comparison
        fig, axes = plt.subplots(nrows=len(valid_sims), ncols=3, figsize=(15, 4.5 * len(valid_sims)),
                                 constrained_layout=True)
        letters = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)']

        for r, (ds, sim_info) in enumerate(zip(datasets, valid_sims)):
            for c, var_cfg in enumerate(VARS_CONFIG):
                ax = axes[r, c]
                data = get_slice_data(ds, var_cfg['var'], TIME_INDEX, SLICE_DIM, SLICE_VAL,
                                      calc_perturbation=var_cfg['pert'])
                plot_panel(ax, var_cfg, data, is_bottom=(r == len(valid_sims) - 1), is_left=(c == 0))

                if r == 0: ax.set_title(var_cfg['title'], fontsize=16, fontweight='bold', pad=15)
                if c == 0: ax.text(-0.35, 0.5, sim_info['title'], transform=ax.transAxes, rotation=90,
                                   va='center', ha='center', fontsize=14, fontweight='bold')
                ax.text(0.05, 0.92, letters[r * 3 + c], transform=ax.transAxes, fontsize=14, fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        save_name = "comparison_plot_grid.png"

    print(f"Saving {save_name}...")
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.show()
    for d in datasets: d.close()


if __name__ == "__main__":
    create_plots()