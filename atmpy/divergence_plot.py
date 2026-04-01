import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ==========================================
# USER CONFIGURATION
# ==========================================
BASE_DIR = "/home/amir/Projects/Python/Atmpy/atmpy/output_data/baroclinic_qg_vortex/_baroclinic_qg_vortex_40_10_40"
FILE_FC = f"{BASE_DIR}_N_10_f_100_QG_off_bgVel_0.nc"
FILE_QG = f"{BASE_DIR}_N_10_f_100_QG_on_bgVel_0.nc"

VAR_NAME = 'p2_nodes'
VAR_LABEL = r"Error in $\pi'$ ($FC - QG$)"

# Coordinate names to look for
COORD_X_REQ = 'x_node'
COORD_Y_REQ = 'y_node'  # Vertical
COORD_Z_REQ = 'z_node'


# ==========================================
# PROCESSING & PLOTTING
# ==========================================

def main():
    path_fc = Path(FILE_FC).expanduser()
    path_qg = Path(FILE_QG).expanduser()

    # 1. Load Data
    print(f"Loading {path_fc.name}...")
    ds1 = xr.open_dataset(path_fc)
    ds2 = xr.open_dataset(path_qg)

    # 2. Detect Coordinates
    # We find the actual names in the file (handling plural/singular cases)
    def get_coord_name(ds, req_name):
        if req_name in ds.coords or req_name in ds.dims: return req_name
        if req_name + 's' in ds.coords: return req_name + 's'
        return req_name  # Fallback

    coord_x = get_coord_name(ds1, COORD_X_REQ)
    coord_y = get_coord_name(ds1, COORD_Y_REQ)
    coord_z = get_coord_name(ds1, COORD_Z_REQ)

    # Identify Time Dimension
    time_dim = 'time'
    if 'time' not in ds1.dims and 't' in ds1.dims:
        time_dim = 't'

    # 3. Synchronize Time Length
    # We slice both to the minimum common length to ensure shapes match
    n_time = min(ds1.sizes[time_dim], ds2.sizes[time_dim])
    ds1 = ds1.isel({time_dim: slice(0, n_time)})
    ds2 = ds2.isel({time_dim: slice(0, n_time)})

    print(f"Processing {n_time} time steps.")

    # 4. Compute Difference (NUMPY MODE)
    # We extract .values to bypass xarray coordinate alignment checks
    # This prevents "RMS Series Shape: (3,)" errors due to float mismatches in time
    data1 = ds1[VAR_NAME].values
    data2 = ds2[VAR_NAME].values

    diff_values = data1 - data2

    # Re-wrap in DataArray for easy plotting (using ds1 coords)
    diff_da = xr.DataArray(diff_values, coords=ds1[VAR_NAME].coords, dims=ds1[VAR_NAME].dims)

    # 5. Calculate RMS Series
    # Identify spatial axes indices for reduction
    # dims are typically (time, x, y, z) -> axes (1, 2, 3)
    spatial_axes = tuple([i for i, d in enumerate(diff_da.dims) if d != time_dim])

    # Compute RMS using numpy to be safe
    rms_series = np.sqrt(np.mean(diff_values ** 2, axis=spatial_axes))

    # Time axis for plotting
    t_plot = ds1[time_dim].values
    if not np.issubdtype(t_plot.dtype, np.number):
        t_plot = np.arange(len(t_plot))

    # ==========================================
    # PLOTTING
    # ==========================================
    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    # Extract Final State for Slices
    final_diff = diff_da.isel({time_dim: -1})
    abs_max = np.max(np.abs(final_diff.values))
    if abs_max == 0: abs_max = 1e-15

    # --- PLOT 1: Vertical Slice (X-Y) ---
    ax1 = fig.add_subplot(gs[0, 0])

    # Slice at mid Z
    iz_mid = final_diff.sizes[coord_z] // 2
    slice_xy = final_diff.isel({coord_z: iz_mid})

    x_vals = slice_xy[coord_x].values
    y_vals = slice_xy[coord_y].values
    data_xy = slice_xy.values.T if slice_xy.shape != (len(y_vals), len(x_vals)) else slice_xy.values

    cf1 = ax1.pcolormesh(x_vals, y_vals, data_xy,
                         cmap='seismic', vmin=-abs_max, vmax=abs_max, shading='auto')
    ax1.set_title(f"Vertical Error Slice ($z_{{mid}}$)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y (vertical)")
    fig.colorbar(cf1, ax=ax1, label=VAR_LABEL)

    # --- PLOT 2: Horizontal Slice (X-Z) ---
    ax2 = fig.add_subplot(gs[0, 1])

    # Slice at mid Y
    iy_mid = final_diff.sizes[coord_y] // 2
    slice_xz = final_diff.isel({coord_y: iy_mid})

    x_vals_h = slice_xz[coord_x].values
    z_vals_h = slice_xz[coord_z].values
    data_xz = slice_xz.values.T if slice_xz.shape != (len(z_vals_h), len(x_vals_h)) else slice_xz.values

    cf2 = ax2.pcolormesh(x_vals_h, z_vals_h, data_xz,
                         cmap='seismic', vmin=-abs_max, vmax=abs_max, shading='auto')
    ax2.set_title(f"Horizontal Error Slice ($y_{{mid}}$)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("z")
    fig.colorbar(cf2, ax=ax2, label=VAR_LABEL)

    # --- PLOT 3: RMS Evolution ---
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(t_plot, rms_series, 'k-', linewidth=2)
    ax3.set_title(f"Evolution of RMS Error (Total Disagreement)")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("RMS Error")
    ax3.grid(True, linestyle='--', alpha=0.5)

    # Mark the slice time
    ax3.plot(t_plot[-1], rms_series[-1], 'ro')

    plt.suptitle(f"Stationary Vortex Comparison\nMax Abs Error: {abs_max:.2e}", fontsize=16, weight='bold')
    plt.show()


if __name__ == "__main__":
    main()