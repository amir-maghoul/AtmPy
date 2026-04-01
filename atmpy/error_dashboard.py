import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

# ==========================================
# 1. USER CONFIGURATION
# ==========================================

# List of files to compare
FILE_PATHS = [
    "/home/amir/Projects/Python/Atmpy/atmpy/output_data/baroclinic_qg_vortex/_baroclinic_qg_vortex_40_10_40_N_10_cor_1000_QG_off_Hfac_4_QGprojection.nc",
    "/home/amir/Projects/Python/Atmpy/atmpy/output_data/baroclinic_qg_vortex/_baroclinic_qg_vortex_40_10_40_N_10_cor_1000_QG_on_Hfac_4_QGprojection.nc",
]

# Custom labels
FILE_LABELS = ["Full Compressible", "Quasi-Geostrophic"]

# Analysis Settings
VAR_NAME = 'u'  # Variable to calculate error for
TIME_INDEX = -1  # Target time step index
SLICE_DIM = 'z'  # Dimension to slice: 'x', 'y', or 'z'
SLICE_VAL = 0.0  # Value to slice at

# Style Constants
UNIT_LABEL = "($10$ km)"
N_LEVELS = 25
ERROR_CMAP = 'cividis'

# Global Styles
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.titlesize': 14,
    'figure.dpi': 150
})


# ==========================================
# 2. DATA PROCESSING
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


def get_slice_data(ds, var_name, t_idx, slice_dim, slice_val):
    if var_name not in ds:
        print(f"Variable {var_name} not found.")
        return None

    da = ds[var_name]

    # Select Time
    if 'time' in da.dims:
        t_max = da.sizes['time'] - 1
        t_sel = t_idx if t_idx >= 0 else t_max + t_idx + 1
        t_sel = np.clip(t_sel, 0, t_max)
        da = da.isel(time=int(t_sel))

    # Slice
    candidates = [d for d in da.dims if slice_dim in d]
    if not candidates: return None
    actual_slice_dim = candidates[0]

    try:
        da_slice = da.sel({actual_slice_dim: slice_val}, method='nearest')
    except:
        return None

    # Axes
    rem_dims = list(da_slice.dims)
    if len(rem_dims) != 2: return None

    sim_dim_v = next((d for d in rem_dims if 'y' in d.lower()), None)
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

    # Transpose if needed
    if rem_dims.index(dim_horiz) < rem_dims.index(dim_vert):
        Z_data = Z_data.T

    return {
        'X': X, 'Y': Y, 'data': Z_data,
        'xlabel': f"{get_plot_label(dim_horiz)} {UNIT_LABEL}",
        'ylabel': f"{get_plot_label(dim_vert)} {UNIT_LABEL}"
    }


def get_error_slice(path, var_name, t_idx, slice_dim, slice_val):
    p = Path(path).expanduser()
    if not p.exists(): return None, f"File missing"

    try:
        with xr.open_dataset(p) as ds:
            d0 = get_slice_data(ds, var_name, 0, slice_dim, slice_val)
            dn = get_slice_data(ds, var_name, t_idx, slice_dim, slice_val)

            if d0 is None or dn is None: return None, "Slice failed"

            # Error Calculation
            error_data = dn['data'] - d0['data']

            result = {
                'X': dn['X'], 'Y': dn['Y'], 'data': error_data,
                'xlabel': dn['xlabel'], 'ylabel': dn['ylabel']
            }
            return result, None
    except Exception as e:
        return None, str(e)


# ==========================================
# 3. PLOTTING (UPDATED)
# ==========================================

def plot_error_field(ax, plot_data, title):
    if plot_data is None:
        ax.text(0.5, 0.5, title, ha='center', va='center')
        return

    X, Y = plot_data['X'], plot_data['Y']
    Z = plot_data['data']

    # Normalize cmap around 0
    max_abs = np.nanmax(np.abs(Z))
    if max_abs == 0: max_abs = 1e-10
    # levels = np.linspace(-max_abs, max_abs, N_LEVELS)
    dmin, dmax = np.nanmin(Z), np.nanmax(Z)
    if np.isclose(dmin, dmax):
        levels = np.linspace(dmin, dmax, N_LEVELS)
    else:
        levels = np.linspace(dmin, dmax, N_LEVELS)
    # ---------------------------------------------------------
    # FORCE SQUARE ASPECT RATIO
    # ---------------------------------------------------------
    ax.set_box_aspect(1)
    # ---------------------------------------------------------

    # 1. Contour
    cf = ax.contourf(X, Y, Z, levels=levels, cmap=ERROR_CMAP, extend='both')

    # 2. Labels
    ax.set_title(title, fontweight='bold', pad=10)
    ax.set_xlabel(plot_data['xlabel'])
    ax.set_ylabel(plot_data['ylabel'])

    # 3. Vertical Colorbar
    # shrink=0.7 matches the square aspect better so the bar isn't taller than the plot
    cbar = plt.colorbar(cf, ax=ax, orientation='vertical', pad=0.04, shrink=0.75)

    # Force scientific notation on the Y-axis (since bar is vertical)
    cbar.ax.ticklabel_format(style='sci', scilimits=(-1, 1), axis='y', useMathText=True)

    # Move the exponent text (e.g. 10^-4) slightly above the bar
    cbar.ax.yaxis.get_offset_text().set_position((0, 1.02))


def create_error_dashboard():
    n_files = len(FILE_PATHS)

    # Increased width slightly (6.5) to accommodate square plots + colorbars nicely
    fig, axes = plt.subplots(1, n_files, figsize=(6.5 * n_files, 6), constrained_layout=True)

    if n_files == 1: axes = [axes]

    print(f"Plotting Error for: {VAR_NAME}")

    for i, path in enumerate(FILE_PATHS):
        ax = axes[i]
        lbl = FILE_LABELS[i] if i < len(FILE_LABELS) else Path(path).name

        result, err = get_error_slice(path, VAR_NAME, TIME_INDEX, SLICE_DIM, SLICE_VAL)

        if result:
            plot_error_field(ax, result, f"{lbl}\nError ($t_{{end}} - t_0$)")
        else:
            ax.text(0.5, 0.5, f"Error:\n{err}", ha='center')

    if VAR_NAME in ["u", "v", "w"]:
        title_var = VAR_NAME
    elif VAR_NAME == "p2_nodes":
        title_var = r"$\pi^{\prime}$"
    else:
        title_var = fr"$\{VAR_NAME}$"
    fig.suptitle(f"Time Evolution Error: {title_var}", fontsize=16, weight='bold')
    plt.show()


if __name__ == "__main__":
    create_error_dashboard()