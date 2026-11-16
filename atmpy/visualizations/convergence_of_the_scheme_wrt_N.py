import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# --- Configuration ---
DATA_FOLDER = "/home/amir/Projects/Python/Atmpy/atmpy/output_data/traveling_vortex_3d/asymptotics/grid_convergence/"
FILE_PREFIX = "_traveling_vortex_3d"
RESOLUTIONS = [15, 20, 30, 40, 60, 80, 100]
VARIABLE_NAME = "rho"


def get_file_path(res):
    filename = f"{FILE_PREFIX}_{res}_{res}_{res}.nc"
    for root, dirs, files in os.walk(DATA_FOLDER):
        if filename in files:
            return os.path.join(root, filename)
    return None


# --- Data Collection ---
errors_l2 = []
errors_linf = []
actual_n = []

for n in RESOLUTIONS:
    path = get_file_path(n)
    if path:
        print(f"Calculating Errors for N = {n} ...")
        with xr.open_dataset(path) as ds:
            # Get variables at the first and last time steps
            initial_state = ds[VARIABLE_NAME].isel(time=0).values
            final_state = ds[VARIABLE_NAME].isel(time=-1).values

            # Error = Final - Initial
            diff = final_state - initial_state

            # Calculate L2 and L_inf norms
            l2_err = np.sqrt(np.mean(diff ** 2))
            linf_err = np.max(np.abs(diff))

            errors_l2.append(l2_err)
            errors_linf.append(linf_err)
            actual_n.append(n)

# Convert to numpy arrays for vector operations
n_array = np.array(actual_n)
l2_array = np.array(errors_l2)
linf_array = np.array(errors_linf)

# --- Plotting ---
# Setup a 1x2 side-by-side subplot (Optimal for standard paper widths)
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

# Paper-standard styling (Grayscale & Colorblind friendly)
# Data: Solid black line with circles
# Reference: Dashed dark-gray line, no markers
data_style = dict(color='k', marker='o', linestyle='-', linewidth=1.5, markersize=6)
ref_style = dict(color='#555555', linestyle='--', linewidth=1.5)

# Group data for clean looping over both subplots
error_data = [
    (axes[0], l2_array, "$L_2$"),
    (axes[1], linf_array, "$L_\\infty$")
]

for ax, err_array, label in error_data:
    # 1. Plot actual simulation data
    ax.loglog(n_array, err_array, label=f'Simulation Error ({label})', **data_style)

    # 2. Add Reference Order Line (2nd Order only)
    first_err = err_array[0]
    first_n = n_array[0]
    offset_2nd = 0.8  # Shifts the theoretical line below the actual data

    line_2nd = (first_err * offset_2nd) * (first_n / n_array) ** 2
    ax.loglog(n_array, line_2nd, label='$\mathcal{O}(N^{-2})$ Reference', **ref_style)

    # 3. Calculate Observed Order (Slope)
    # Using the mid-to-high resolution range for a more stable/accurate slope
    slope, _ = np.polyfit(np.log(n_array[2:]), np.log(err_array[2:]), 1)
    observed_order = abs(slope)

    # 4. Formatting and aesthetics
    ax.set_title(f"{label} Error Convergence", fontsize=14, pad=12)
    ax.set_xlabel("Grid Resolution ($N$)", fontsize=14)
    ax.set_ylabel(f"{label} Error Norm", fontsize=14)

    # Ticks and Grid
    ax.tick_params(axis='both', which='both', labelsize=13)
    ax.grid(True, which="major", linestyle="-", alpha=0.3)
    ax.grid(True, which="minor", linestyle=":", alpha=0.15)

    # Legend without a box frame (cleaner for papers)
    ax.legend(fontsize=13, frameon=False, loc='upper right')

    # Annotate the observed order cleanly in the bottom left
    # Using transAxes keeps it relatively pinned to the bottom left corner regardless of values
    ax.text(0.05, 0.05, f"Observed Order: {observed_order:.2f}",
            transform=ax.transAxes, fontsize=13,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

# # Super-title for the entire figure
# plt.suptitle(f"3D Grid Convergence: Final vs. Initial State ({VARIABLE_NAME})",
#              fontsize=15, fontweight='bold', y=1.02)

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.show()