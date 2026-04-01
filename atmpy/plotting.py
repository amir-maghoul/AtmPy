import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

# ==========================================
# 1. USER CONFIGURATION
# ==========================================
FILE_PATH = "/home/amir/Projects/Python/Atmpy/atmpy/output_data/baroclinic_qg_vortex/_baroclinic_qg_vortex_40_10_40_N_10_cor_1000_QG_on_Hfac_4_QGprojection.nc"
# FILE_PATH = "/home/amir/Projects/Python/Atmpy/atmpy/output_data/baroclinic_qg_vortex/_baroclinic_qg_vortex_40_10_40.nc"

# --- Visual Configuration ---
# 1. Color Blind Friendliness Switch
# True: Uses 'cividis' (seq) and 'PuOr' (div). False: Uses 'viridis' (seq) and 'RdBu_r' (div).
USE_COLORBLIND_SAFE = True

# 2. Layout Configuration
# 'grid': 2x2 Square | 'column': 4x1 Vertical Stack
LAYOUT_MODE = 'grid'

# 3. Plotting Selection
TIME_INDEX = 0
SLICE_DIM = 'z'
SLICE_VAL = 0.0

# Toggle Perturbation Plotting (Calculates Field - Mean)
PLOT_PERTURBATION = True

# Constants
# Set to None to automatically normalize by the final time step (t_end = 1.0)
# Set to a number (e.g. 86400) to normalize by that specific value.
T_REF = None

# Style Constants
UNIT_LABEL = "($10$ km)"
N_LEVELS = 25
LINE_WIDTH = 0.2
LINE_COLOR = 'k'

# Global Matplotlib Styles
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 150
})

# ==========================================
# 2. COLOR & VARIABLE SETUP
# ==========================================

if USE_COLORBLIND_SAFE:
    CMAP_SEQ = 'cividis'
    CMAP_DIV = 'PuOr'
    CMAP_DIV = 'cividis'

else:
    CMAP_SEQ = 'viridis'
    CMAP_DIV = 'RdBu_r'

VAR_SPECS = {
    'p2_nodes': {'title': r"$\pi'$ (Exner Pressure)", 'type': 'diverging'},
    'rho': {'title': r"$\rho$ (Density)", 'type': 'sequential'},
    'Y': {'title': r"$\Theta$ (Pot. Temp)", 'type': 'sequential'},
    'u': {'title': r"$u$ (Velocity)", 'type': 'diverging'}
}


# ==========================================
# 3. DATA PROCESSING
# ==========================================

def get_plot_label(sim_dim_name):
    """Maps simulation dimensions to plot labels."""
    clean_name = sim_dim_name.replace('_nodes', '').replace('_node', '').lower()
    if 'y' in clean_name:
        return 'z'
    elif 'z' in clean_name:
        return 'y'
    elif 'x' in clean_name:
        return 'x'
    return clean_name


def calculate_perturbation(da):
    """Calculates perturbation by subtracting the horizontal mean."""
    horiz_dims = [d for d in da.dims if 'x' in d.lower() or 'z' in d.lower()]
    if not horiz_dims: return da
    da_bg = da.mean(dim=horiz_dims)
    return da - da_bg


def get_slice_data(ds, var_name, t_idx, slice_dim, slice_val, calc_perturbation=False):
    """Extracts a 2D slice, handles mapping, optionally calculates perturbation."""
    if var_name not in ds: return None
    da = ds[var_name]

    if 'time' in da.dims:
        t_idx = min(t_idx, da.sizes['time'] - 1)
        da = da.isel(time=t_idx)

    if calc_perturbation:
        da = calculate_perturbation(da)

    candidates = [d for d in da.dims if slice_dim in d]
    if not candidates: return None
    actual_slice_dim = candidates[0]

    try:
        da_slice = da.sel({actual_slice_dim: slice_val}, method='nearest')
    except:
        return None

    rem_dims = list(da_slice.dims)
    if len(rem_dims) != 2: return None

    # Identify Vertical vs Horizontal
    sim_dim_v = next((d for d in rem_dims if 'y' in d.lower()), None)  # Sim Y is Vert

    if sim_dim_v:
        dim_vert = sim_dim_v
        dim_horiz = [d for d in rem_dims if d != dim_vert][0]
    else:
        sim_dim_z = next((d for d in rem_dims if 'z' in d.lower()), None)
        if sim_dim_z:
            dim_vert = sim_dim_z
            dim_horiz = [d for d in rem_dims if d != dim_vert][0]
        else:
            dim_horiz, dim_vert = rem_dims[1], rem_dims[0]

    X = da_slice[dim_horiz].values
    Y = da_slice[dim_vert].values
    Z_data = da_slice.values

    # Transpose so Horizontal is X-axis
    idx_h = rem_dims.index(dim_horiz)
    idx_v = rem_dims.index(dim_vert)
    if idx_h < idx_v:
        Z_data = Z_data.T

    return {
        'X': X, 'Y': Y, 'data': Z_data,
        'xlabel': f"{get_plot_label(dim_horiz)} {UNIT_LABEL}",
        'ylabel': f"{get_plot_label(dim_vert)} {UNIT_LABEL}"
    }


def get_time_float(t_val):
    """Robustly converts xarray time to float."""
    if hasattr(t_val, 'dtype'):
        if np.issubdtype(t_val.dtype, np.timedelta64) or np.issubdtype(t_val.dtype, np.datetime64):
            return t_val.astype('float64') / 1e9
    return float(t_val)


# ==========================================
# 4. PLOTTING
# ==========================================

def plot_variable(ax, spec, plot_data, is_perturbation):
    if plot_data is None:
        ax.text(0.5, 0.5, "Data Missing", ha='center', va='center')
        ax.set_box_aspect(1)
        return

    X, Y = plot_data['X'], plot_data['Y']
    Z = plot_data['data']
    dmin, dmax = np.nanmin(Z), np.nanmax(Z)

    # 1. Determine Colormap
    if is_perturbation or spec['type'] == 'diverging':
        cmap_to_use = CMAP_DIV
    else:
        cmap_to_use = CMAP_SEQ

    # 2. Determine Levels
    # CHANGED: Always use Min to Max levels, even for perturbations.
    if np.isclose(dmin, dmax):
        levels = np.linspace(dmin - 0.1, dmax + 0.1, N_LEVELS)
    else:
        levels = np.linspace(dmin, dmax, N_LEVELS)

    # Filled Contour
    cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap_to_use, extend='both')

    # Line Contour
    if not np.isclose(dmin, dmax):
        ax.contour(X, Y, Z, levels=levels, colors=LINE_COLOR, linewidths=LINE_WIDTH)

    # Style
    ax.set_title(spec['title'], fontweight='bold', pad=10)
    ax.set_xlabel(plot_data['xlabel'])
    ax.set_ylabel(plot_data['ylabel'])

    # FORCE SQUARE ASPECT RATIO
    ax.set_box_aspect(1)
    ax.set_aspect('auto')

    # Colorbar
    cbar = plt.colorbar(cf, ax=ax, orientation='vertical', shrink=0.8)
    cbar.ax.ticklabel_format(style='sci', scilimits=(-1, 1), axis='y', useMathText=True)


def create_dashboard():
    path = Path(FILE_PATH).expanduser()
    if not path.exists():
        print(f"Error: File not found at {path}")
        return

    try:
        ds = xr.open_dataset(path)
    except Exception as e:
        print(f"Error opening file: {e}")
        return

    # --- Layout Logic ---
    if LAYOUT_MODE == 'column':
        nrows, ncols = 4, 1
        figsize = (6, 22)
    else:
        nrows, ncols = 2, 2
        figsize = (12, 12)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    if nrows * ncols > 1:
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]

    # --- Time Normalization Logic ---
    try:
        # Get current time
        t_current_raw = ds['time'].isel(time=TIME_INDEX).values
        t_current = get_time_float(t_current_raw)

        # Get Ref time
        if T_REF is None:
            t_ref_raw = ds['time'].max().values
            t_ref = get_time_float(t_ref_raw)
            if t_ref == 0: t_ref = 1.0
        else:
            t_ref = float(T_REF)

        t_norm = t_current / t_ref
    except Exception as e:
        print(f"Time calculation warning: {e}")
        t_norm = float(TIME_INDEX)

    # Main Title
    mode_str = "Perturbation" if PLOT_PERTURBATION else "Full State"
    fig.suptitle(f"{mode_str} | Slice: {get_plot_label(SLICE_DIM)}={SLICE_VAL:.2f}\n($t / t_{{end}} = {t_norm:.2f}$)",
                 fontsize=20, weight='bold')

    # Plot Loop
    var_names = list(VAR_SPECS.keys())
    for i, ax in enumerate(axes_flat):
        if i < len(var_names):
            vname = var_names[i]
            spec = VAR_SPECS[vname]

            calc_pert = False
            if PLOT_PERTURBATION and vname in ['rho', 'Y']:
                calc_pert = True

            pdata = get_slice_data(ds, vname, TIME_INDEX, SLICE_DIM, SLICE_VAL, calc_perturbation=calc_pert)
            plot_variable(ax, spec, pdata, is_perturbation=calc_pert)
        else:
            ax.axis('off')

    plt.show()
    ds.close()


if __name__ == "__main__":
    create_dashboard()

# import xarray as xr
# import matplotlib.pyplot as plt
# import matplotlib.ticker as mticker
# import numpy as np
# from pathlib import Path
#
# # ==========================================
# # 1. USER CONFIGURATION
# # ==========================================
# FILE_PATH = "/home/amir/Projects/Python/Atmpy/atmpy/output_data/baroclinic_qg_vortex/_baroclinic_qg_vortex_40_10_40_N_10_cor_1000_QG_on_Hfac_4_QGprojection.nc"
# # FILE_PATH = "/home/amir/Projects/Python/Atmpy/atmpy/output_data/baroclinic_qg_vortex/_baroclinic_qg_vortex_40_10_40.nc"
#
# # --- Visual Configuration ---
# # 1. Color Blind Friendliness Switch
# # True: Uses 'cividis' (seq) and 'PuOr' (div). False: Uses 'viridis' (seq) and 'RdBu_r' (div).
# USE_COLORBLIND_SAFE = True
#
# # 2. Layout Configuration
# # 'grid': 2x2 Square | 'column': 4x1 Vertical Stack
# LAYOUT_MODE = 'grid'
#
# # 3. Plotting Selection
# TIME_INDEX = 0
# SLICE_DIM = 'y'
# SLICE_VAL = 0.0
#
# # Toggle Perturbation Plotting (Calculates Field - Mean)
# PLOT_PERTURBATION = True
#
# # Constants
# # Set to None to automatically normalize by the final time step (t_end = 1.0)
# # Set to a number (e.g. 86400) to normalize by that specific value.
# T_REF = None
#
# # Style Constants
# UNIT_LABEL = "($10$ km)"
# N_LEVELS = 25
# LINE_WIDTH = 0.2
# LINE_COLOR = 'k'
#
# # Global Matplotlib Styles
# plt.rcParams.update({
#     'font.family': 'serif',
#     'font.size': 14,
#     'axes.titlesize': 16,
#     'axes.labelsize': 14,
#     'xtick.labelsize': 12,
#     'ytick.labelsize': 12,
#     'figure.dpi': 150
# })
#
# # ==========================================
# # 2. COLOR & VARIABLE SETUP
# # ==========================================
#
# if USE_COLORBLIND_SAFE:
#     CMAP_SEQ = 'cividis'
#     CMAP_DIV = 'PuOr'
#     CMAP_DIV = 'cividis'
#
# else:
#     CMAP_SEQ = 'viridis'
#     CMAP_DIV = 'RdBu_r'
#
# VAR_SPECS = {
#     'p2_nodes': {'title': r"$\pi'$ (Exner Pressure)", 'type': 'diverging'},
#     'rho': {'title': r"$\rho$ (Density)", 'type': 'sequential'},
#     'Y': {'title': r"$\Theta$ (Pot. Temp)", 'type': 'sequential'},
#     'u': {'title': r"$u$ (Velocity)", 'type': 'diverging'}
# }
#
#
# # ==========================================
# # 3. DATA PROCESSING
# # ==========================================
#
# def get_plot_label(sim_dim_name):
#     """Maps simulation dimensions to plot labels."""
#     clean_name = sim_dim_name.replace('_nodes', '').replace('_node', '').lower()
#     if 'y' in clean_name:
#         return 'z'
#     elif 'z' in clean_name:
#         return 'y'
#     elif 'x' in clean_name:
#         return 'x'
#     return clean_name
#
#
# def calculate_perturbation(da):
#     """Calculates perturbation by subtracting the horizontal mean."""
#     horiz_dims = [d for d in da.dims if 'x' in d.lower() or 'z' in d.lower()]
#     if not horiz_dims: return da
#     da_bg = da.mean(dim=horiz_dims)
#     return da - da_bg
#
#
# def get_slice_data(ds, var_name, t_idx, slice_dim, slice_val, calc_perturbation=False):
#     """Extracts a 2D slice, handles mapping, optionally calculates perturbation."""
#     if var_name not in ds: return None
#     da = ds[var_name]
#
#     if 'time' in da.dims:
#         t_idx = min(t_idx, da.sizes['time'] - 1)
#         da = da.isel(time=t_idx)
#
#     if calc_perturbation:
#         da = calculate_perturbation(da)
#
#     candidates = [d for d in da.dims if slice_dim in d]
#     if not candidates: return None
#     actual_slice_dim = candidates[0]
#
#     try:
#         da_slice = da.sel({actual_slice_dim: slice_val}, method='nearest')
#     except:
#         return None
#
#     rem_dims = list(da_slice.dims)
#     if len(rem_dims) != 2: return None
#
#     # Identify Vertical vs Horizontal
#     sim_dim_v = next((d for d in rem_dims if 'y' in d.lower()), None)  # Sim Y is Vert
#
#     if sim_dim_v:
#         dim_vert = sim_dim_v
#         dim_horiz = [d for d in rem_dims if d != dim_vert][0]
#     else:
#         sim_dim_z = next((d for d in rem_dims if 'z' in d.lower()), None)
#         if sim_dim_z:
#             dim_vert = sim_dim_z
#             dim_horiz = [d for d in rem_dims if d != dim_vert][0]
#         else:
#             dim_horiz, dim_vert = rem_dims[1], rem_dims[0]
#
#     X = da_slice[dim_horiz].values
#     Y = da_slice[dim_vert].values
#     Z_data = da_slice.values
#
#     # Transpose so Horizontal is X-axis
#     idx_h = rem_dims.index(dim_horiz)
#     idx_v = rem_dims.index(dim_vert)
#     if idx_h < idx_v:
#         Z_data = Z_data.T
#
#     return {
#         'X': X, 'Y': Y, 'data': Z_data,
#         'xlabel': f"{get_plot_label(dim_horiz)} {UNIT_LABEL}",
#         'ylabel': f"{get_plot_label(dim_vert)} {UNIT_LABEL}"
#     }
#
#
# def get_time_float(t_val):
#     """Robustly converts xarray time to float."""
#     if hasattr(t_val, 'dtype'):
#         if np.issubdtype(t_val.dtype, np.timedelta64) or np.issubdtype(t_val.dtype, np.datetime64):
#             return t_val.astype('float64') / 1e9
#     return float(t_val)
#
#
# # ==========================================
# # 4. PLOTTING
# # ==========================================
#
# def plot_variable(ax, spec, plot_data, is_perturbation):
#     if plot_data is None:
#         ax.text(0.5, 0.5, "Data Missing", ha='center', va='center')
#         ax.set_box_aspect(1)
#         return
#
#     X, Y = plot_data['X'], plot_data['Y']
#     Z = plot_data['data']
#     dmin, dmax = np.nanmin(Z), np.nanmax(Z)
#
#     # 1. Determine Colormap
#     if is_perturbation or spec['type'] == 'diverging':
#         cmap_to_use = CMAP_DIV
#     else:
#         cmap_to_use = CMAP_SEQ
#
#     # 2. Determine Levels
#     # Only force symmetric levels (centering 0 at white) if it is a Perturbation.
#     if is_perturbation:
#         limit = max(abs(dmin), abs(dmax))
#         if limit == 0: limit = 1e-10
#         levels = np.linspace(-limit, limit, N_LEVELS)
#         cmap_to_use = CMAP_DIV
#     else:
#         if np.isclose(dmin, dmax):
#             levels = np.linspace(dmin - 0.1, dmax + 0.1, N_LEVELS)
#         else:
#             levels = np.linspace(dmin, dmax, N_LEVELS)
#
#     # Filled Contour
#     cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap_to_use, extend='both')
#
#     # Line Contour
#     if not np.isclose(dmin, dmax):
#         ax.contour(X, Y, Z, levels=levels, colors=LINE_COLOR, linewidths=LINE_WIDTH)
#
#     # Style
#     ax.set_title(spec['title'], fontweight='bold', pad=10)
#     ax.set_xlabel(plot_data['xlabel'])
#     ax.set_ylabel(plot_data['ylabel'])
#
#     # FORCE SQUARE ASPECT RATIO
#     ax.set_box_aspect(1)
#     ax.set_aspect('auto')
#
#     # Colorbar
#     cbar = plt.colorbar(cf, ax=ax, orientation='vertical', shrink=0.8)
#     cbar.ax.ticklabel_format(style='sci', scilimits=(-1, 1), axis='y', useMathText=True)
#
#
# def create_dashboard():
#     path = Path(FILE_PATH).expanduser()
#     if not path.exists():
#         print(f"Error: File not found at {path}")
#         return
#
#     try:
#         ds = xr.open_dataset(path)
#     except Exception as e:
#         print(f"Error opening file: {e}")
#         return
#
#     # --- Layout Logic ---
#     if LAYOUT_MODE == 'column':
#         nrows, ncols = 4, 1
#         figsize = (6, 22)
#     else:
#         nrows, ncols = 2, 2
#         figsize = (12, 12)
#
#     fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
#     if nrows * ncols > 1:
#         axes_flat = axes.flatten()
#     else:
#         axes_flat = [axes]
#
#     # --- Time Normalization Logic ---
#     try:
#         # Get current time
#         t_current_raw = ds['time'].isel(time=TIME_INDEX).values
#         t_current = get_time_float(t_current_raw)
#
#         # Get Ref time
#         if T_REF is None:
#             t_ref_raw = ds['time'].max().values
#             t_ref = get_time_float(t_ref_raw)
#             if t_ref == 0: t_ref = 1.0
#         else:
#             t_ref = float(T_REF)
#
#         t_norm = t_current / t_ref
#     except Exception as e:
#         print(f"Time calculation warning: {e}")
#         t_norm = float(TIME_INDEX)
#
#     # Main Title
#     mode_str = "Perturbation" if PLOT_PERTURBATION else "Full State"
#     fig.suptitle(f"{mode_str} | Slice: {get_plot_label(SLICE_DIM)}={SLICE_VAL:.2f}\n($t / t_{{end}} = {t_norm:.2f}$)",
#                  fontsize=20, weight='bold')
#
#     # Plot Loop
#     var_names = list(VAR_SPECS.keys())
#     for i, ax in enumerate(axes_flat):
#         if i < len(var_names):
#             vname = var_names[i]
#             spec = VAR_SPECS[vname]
#
#             calc_pert = False
#             if PLOT_PERTURBATION and vname in ['rho', 'Y']:
#                 calc_pert = True
#
#             pdata = get_slice_data(ds, vname, TIME_INDEX, SLICE_DIM, SLICE_VAL, calc_perturbation=calc_pert)
#             plot_variable(ax, spec, pdata, is_perturbation=calc_pert)
#         else:
#             ax.axis('off')
#
#     plt.show()
#     ds.close()
#
#
# if __name__ == "__main__":
#     create_dashboard()