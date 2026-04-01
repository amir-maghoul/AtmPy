
DATA_FOLDER = "/home/amir/Projects/Python/Atmpy/atmpy/output_data/traveling_vortex_3d/asymptotics/"

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re
from matplotlib.ticker import LogLocator, LogFormatterMathtext

# ==========================================
# USER CONFIGURATION
# ==========================================
DATA_FOLDER = "/home/amir/Projects/Python/Atmpy/atmpy/output_data/traveling_vortex_3d/asymptotics/80_3_80_full/"
# DATA_FOLDER = "/home/amir/Projects/Python/Atmpy/atmpy/output_data/baroclinic_qg_vortex/asymptotics/40_10_40_full/"


VAR_NAME = 'p2_nodes'
VAR_LABEL = r"$\pi'$ (Exner Pressure)"

# Regex for 'cor'
PARAM_PATTERN = r"cor_(\d+)"
PARAM_LABEL = "Coriolis Factor"

# Styling
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'lines.linewidth': 2,
    'lines.markersize': 8,
    'figure.dpi': 150
})


# ==========================================
# FUNCTIONS
# ==========================================

def get_parameter_from_filename(filename):
    match = re.search(PARAM_PATTERN, filename)
    if match:
        return int(match.group(1))
    return None


def compute_relative_errors(file_path, var_name):
    """
    Computes RELATIVE L1, L2, and Linf errors.
    """
    try:
        with xr.open_dataset(file_path) as ds:
            if var_name not in ds:
                return None
            data = ds[var_name]
            if 'time' not in data.dims or data.sizes['time'] < 2:
                return None

            state_0 = data.isel(time=0).values
            state_end = data.isel(time=-1).values

            # Normalize by initial state
            denom_l1 = np.mean(np.abs(state_0.flatten()))
            denom_l2 = np.sqrt(np.mean(state_0.flatten() ** 2))
            denom_linf = np.max(np.abs(state_0.flatten()))

            # Avoid division by zero
            if denom_l2 < 1e-15:
                return None

            # Absolute diff
            diff = state_end - state_0
            diff_flat = diff.flatten()
            abs_diff = np.abs(diff_flat)

            # Relative errors
            rel_l1 = np.mean(abs_diff) / denom_l1
            rel_l2 = np.sqrt(np.mean(diff_flat ** 2)) / denom_l2
            rel_linf = np.max(abs_diff) / denom_linf

            return rel_l1, rel_l2, rel_linf

    except Exception as e:
        print(f"  [Error] {file_path.name}: {e}")
        return None


# ==========================================
# MAIN
# ==========================================

def main():
    folder = Path(DATA_FOLDER).expanduser().resolve()
    if not folder.is_dir():
        print(f"Error: Folder not found: {folder}")
        return

    files = list(folder.glob("*.nc"))
    if not files:
        print("No .nc files found.")
        return

    results = []
    print(f"Found {len(files)} files. Processing...")

    for f in files:
        param = get_parameter_from_filename(f.name)
        if param is not None:
            errors = compute_relative_errors(f, VAR_NAME)
            if errors:
                l1, l2, linf = errors
                results.append({'param': param, 'l1': l1, 'l2': l2, 'linf': linf})
                print(f"  [OK] cor={param}: Rel L2={l2:.2e}")

    if not results:
        print("No valid results.")
        return

    results.sort(key=lambda x: x['param'])

    x_vals = [r['param'] for r in results]
    l1_vals = [r['l1'] for r in results]
    l2_vals = [r['l2'] for r in results]
    linf_vals = [r['linf'] for r in results]

    # --- PLOTTING ---
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)

    cmap = plt.get_cmap('cividis')
    colors = cmap(np.linspace(0, 0.9, 3))

    ax.plot(x_vals, l1_vals, marker='o', linestyle='-', color=colors[0], label=r'Relative $L_1$')
    ax.plot(x_vals, l2_vals, marker='s', linestyle='--', color=colors[1], label=r'Relative $L_2$')
    ax.plot(x_vals, linf_vals, marker='^', linestyle=':', color=colors[2], label=r'Relative $L_\infty$')

    ax.set_title(f"Relative Error Evolution: {VAR_LABEL}", fontweight='bold')
    ax.set_xlabel(PARAM_LABEL)
    ax.set_ylabel(r"Relative Error") # ($\frac{||\Delta||}{||State_0||}$)

    # 1. Enable Log Scales
    ax.set_xscale('log')
    ax.set_yscale('log')

    # 2. Standard Log Formatting (Powers of 10)
    # This replaces the custom "1, 10, 25..." list with standard log ticks (10^0, 10^1...)
    # base=10.0 ensures ticks are at powers of 10
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(labelOnlyBase=False))

    # Grid (Major = Solid, Minor = Dotted)
    ax.grid(True, which="major", ls="-", alpha=0.5)
    ax.grid(True, which="minor", ls=":", alpha=0.3)

    ax.legend()

    plt.show()


if __name__ == "__main__":
    main()