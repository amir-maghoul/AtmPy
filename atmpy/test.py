import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import logging
from typing import Optional, Dict

# --- (Configuration and plot_slice_on_ax function remain the same as before) ---
# --- I am including them here for a complete, runnable script ---

# --- 1. Configuration ---
FILE_PATH_1 = '/home/amir/Projects/Python/Atmpy/atmpy/output_data/zonal_jet_stream/_zonal_jet_stream_80_3_80_comp_no_filter.nc'
FILE_PATH_2 = '/home/amir/Projects/Python/Atmpy/atmpy/output_data/zonal_jet_stream/_zonal_jet_stream_80_3_80_comp_filter_2_damping_0.1.nc'
VARIABLES_TO_PLOT = ['u', 'p2_nodes', 'w']
SUBPLOT_TITLES = {
    'u': r'$u$',
    'p2_nodes': r'$\pi$',
    'w': r'$v$'
}
SLICE_DIMENSION = 'y'
SLICE_VALUE = 0.0
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- 2. Plotting Function (Unchanged) ---
def plot_slice_on_ax(
        ds: xr.Dataset, variable_name: str, time_index: int, slice_dim: str,
        slice_value: float, ax: plt.Axes, title: str, cmap: str = 'viridis',
        symmetric_cbar: bool = False,
) -> None:
    # This function is identical to the previous version
    if variable_name not in ds:
        logger.error(f"Variable '{variable_name}' not found.")
        ax.text(0.5, 0.5, f'"{variable_name}"\nnot found', ha='center', va='center')
        return
    data_var = ds[variable_name]
    data_at_time = data_var.isel(time=time_index)
    is_nodal = any("node" in d for d in data_var.dims)
    slice_dim_actual = f"{slice_dim}_node" if is_nodal and f"{slice_dim}_node" in data_at_time.dims else slice_dim
    if slice_dim_actual not in data_at_time.dims: return
    plot_data_2d = data_at_time.sel({slice_dim_actual: slice_value}, method="nearest")
    plot_dims = [dim for dim in ["x", "y", "z"] if dim != slice_dim]
    x_dim_name, y_dim_name = ([f"{dim}_node" if is_nodal and f"{dim}_node" in ds.coords else dim for dim in plot_dims])
    x_coords, y_coords = ds[x_dim_name].values, ds[y_dim_name].values
    plot_data_for_contour = plot_data_2d.values
    if plot_data_2d.dims.index(y_dim_name) == 1: plot_data_for_contour = plot_data_for_contour.T
    data_min, data_max = np.nanmin(plot_data_for_contour), np.nanmax(plot_data_for_contour)
    if symmetric_cbar:
        abs_max = max(abs(data_min), abs(data_max))
        data_min, data_max = -abs_max, abs_max
    levels = np.linspace(data_min, data_max, 31) if not np.isclose(data_min, data_max) else 31
    contour = ax.contourf(x_coords, y_coords, plot_data_for_contour, cmap=cmap, levels=levels)
    ax.contour(x_coords, y_coords, plot_data_for_contour, colors="k", linewidths=0.2, levels=levels)
    cbar = ax.figure.colorbar(contour, ax=ax, orientation="vertical", pad=0.02)
    # cbar.set_label(f"({data_var.attrs.get('units', '')})")
    default_ticks = cbar.get_ticks()
    all_ticks = np.unique(np.concatenate([default_ticks, [data_min, data_max]]))
    cbar.set_ticks(all_ticks)
    cbar.ax.set_yticklabels([f"{t:.6f}" for t in all_ticks])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title, fontsize=18)


# --- 3. Main Script with 2+1 Layout ---

def main():
    try:
        ds1 = xr.open_dataset(FILE_PATH_1)
        ds2 = xr.open_dataset(FILE_PATH_2)
    except FileNotFoundError as e:
        logger.error(f"Could not open file: {e}. Please check file paths.")
        return

    figure_definitions = [
        {'title': 'Initial Values', 'ds': ds1, 'time_idx': 0},
        {'title': 'Final State at t=10000s (No Filter)', 'ds': ds1, 'time_idx': -1},
        {'title': 'Final State at t=10000s (Filtered)', 'ds': ds2, 'time_idx': -1},
    ]

    for fig_def in figure_definitions:
        # --- NEW: Define the 2+1 layout using subplot_mosaic ---
        # 'u' and 'pi' are on the top row, 'w' spans the entire bottom row.
        layout = [
            ["u_ax", "pi_ax"],
            ["w_ax", "w_ax"],
        ]

        fig, axes_dict = plt.subplot_mosaic(layout, figsize=(12, 8), sharex=True)
        fig.suptitle(fig_def['title'], fontsize=18)

        # --- Plot u ---
        plot_slice_on_ax(
            ds=fig_def['ds'], variable_name='u', time_index=fig_def['time_idx'],
            slice_dim=SLICE_DIMENSION, slice_value=SLICE_VALUE,
            ax=axes_dict['u_ax'], title=SUBPLOT_TITLES['u']
        )

        # --- Plot pi ---
        plot_slice_on_ax(
            ds=fig_def['ds'], variable_name='p2_nodes', time_index=fig_def['time_idx'],
            slice_dim=SLICE_DIMENSION, slice_value=SLICE_VALUE,
            ax=axes_dict['pi_ax'], title=SUBPLOT_TITLES['p2_nodes']
        )

        # --- Plot w (v) ---
        plot_slice_on_ax(
            ds=fig_def['ds'], variable_name='w', time_index=fig_def['time_idx'],
            slice_dim=SLICE_DIMENSION, slice_value=SLICE_VALUE,
            ax=axes_dict['w_ax'], title=SUBPLOT_TITLES['w']
        )

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.show()

    ds1.close()
    ds2.close()


if __name__ == '__main__':
    main()