import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path
import re

# ==========================================
# USER CONFIGURATION
# ==========================================
DATA_FOLDER = "/home/amir/Projects/Python/Atmpy/atmpy/output_data/baroclinic_qg_vortex/"

# Variable to analyze
VAR_NAME = 'p2_nodes'
VAR_LABEL = r"$\pi'$ Difference (Ageostrophic component)"

# Regex to extract the parameter 'f'
PARAM_PATTERN = r"_f_([\d\.]+)_"
PARAM_LABEL = r"Coriolis Parameter ($f$)"

# File Suffixes
FC_SUFFIX = "_fc.nc"
QG_SUFFIX = "_qg.nc"

# Plot Styling
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
# LOGIC
# ==========================================

def get_param(filename):
    match = re.search(PARAM_PATTERN, filename)
    return float(match.group(1)) if match else None


def find_pairs(folder):
    folder = Path(folder).expanduser().resolve()
    files = list(folder.glob("*.nc"))

    pairs = {}

    for f in files:
        f_val = get_param(f.name)
        if f_val is None: continue

        if f_val not in pairs:
            pairs[f_val] = {'FC': None, 'QG': None}

        if f.name.endswith(FC_SUFFIX):
            pairs[f_val]['FC'] = f
        elif f.name.endswith(QG_SUFFIX):
            pairs[f_val]['QG'] = f

    # Return only complete pairs
    return {k: v for k, v in pairs.items() if v['FC'] and v['QG']}


def compute_diff_norm(fc_path, qg_path, var_name):
    try:
        with xr.open_dataset(fc_path) as ds_fc, xr.open_dataset(qg_path) as ds_qg:
            if var_name not in ds_fc or var_name not in ds_qg:
                return None

            # Get final time step
            data_fc = ds_fc[var_name].isel(time=-1).values
            data_qg = ds_qg[var_name].isel(time=-1).values

            if data_fc.shape != data_qg.shape:
                print(f"Shape mismatch at f={get_param(fc_path.name)}")
                return None

            diff = (data_fc - data_qg).flatten()

            # L2 and Linf Norms
            l2 = np.sqrt(np.mean(diff ** 2))
            linf = np.max(np.abs(diff))

            return l2, linf

    except Exception as e:
        print(f"Error: {e}")
        return None


# ==========================================
# MAIN
# ==========================================

def main():
    folder = Path(DATA_FOLDER).expanduser().resolve()
    if not folder.is_dir():
        print(f"Error: Folder not found: {folder}")
        return

    pairs = find_pairs(folder)

    if not pairs:
        print("No complete pairs found.")
        return

    print(f"Found {len(pairs)} pairs. Calculating...")

    results = []
    for f_val, paths in pairs.items():
        norms = compute_diff_norm(paths['FC'], paths['QG'], VAR_NAME)
        if norms:
            l2, linf = norms
            results.append((f_val, l2, linf))
            print(f"  f={f_val: <6} | L2={l2:.2e} | Linf={linf:.2e}")

    results.sort(key=lambda x: x[0])

    f_vals = [r[0] for r in results]
    l2_vals = [r[1] for r in results]
    linf_vals = [r[2] for r in results]

    # --- PLOTTING ---
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    # Plot Lines
    ax.plot(f_vals, l2_vals, 'o-', color='tab:blue', label=r'$L_2$ Norm (RMS)')
    ax.plot(f_vals, linf_vals, 's--', color='tab:red', label=r'$L_\infty$ Norm (Max)')

    # Theoretical Slope Reference (1/f)
    if len(f_vals) > 1:
        f_arr = np.array(f_vals)
        ref_y = l2_vals[0] * (f_arr[0] / f_arr) ** 1
        ax.plot(f_arr, ref_y, 'k:', alpha=0.6, label=r'Ref: $O(f^{-1})$')

    # --- AXIS SCALING (LOG-LOG) ---
    ax.set_xscale('log')  # <--- LOGARITHMIC X-AXIS
    ax.set_yscale('log')  # <--- LOGARITHMIC Y-AXIS

    # Format X-axis to show regular numbers (10, 100) instead of 10^1, 10^2
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())

    ax.set_xlabel(PARAM_LABEL)
    ax.set_ylabel(f"Difference Magnitude\n({VAR_LABEL})")
    ax.set_title(f"Convergence of Full Compressible to QG\n(Log-Log Plot)")

    ax.grid(True, which="major", ls="-", alpha=0.4)
    ax.grid(True, which="minor", ls=":", alpha=0.2)
    ax.legend()

    plt.show()


if __name__ == "__main__":
    main()