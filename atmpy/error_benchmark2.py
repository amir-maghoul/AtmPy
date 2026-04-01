import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re
from matplotlib.ticker import LogLocator, LogFormatterMathtext

# ==========================================
# 1. USER CONFIGURATION
# ==========================================

# Path to the "Reference" simulations (Sim A)
FOLDER_B = "/home/amir/Projects/Python/Atmpy/atmpy/output_data/traveling_vortex_3d/asymptotics/80_3_80_full/redundant2/"

# Path to the "Comparison" simulations (Sim B)
FOLDER_A = "/home/amir/Projects/Python/Atmpy/atmpy/output_data/traveling_vortex_3d/asymptotics/80_3_80_qg/redundant2/"

# Variable to analyze
VAR_NAME = 'p2_nodes'
VAR_LABEL = r"$\pi'$ (Exner Pressure)"

# Parameter Extraction Configuration
PARAM_PATTERN = r"cor_(\d+)"
PARAM_LABEL = "Coriolis Factor"

# Time Selection (-1 for end time)
TIME_INDEX = -1

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
# 2. FUNCTIONS
# ==========================================

def get_parameter_from_filename(filename):
    """Extracts the integer parameter from the filename using regex."""
    match = re.search(PARAM_PATTERN, filename)
    if match:
        return int(match.group(1))
    return None


def compute_difference_metrics(file_a, file_b, var_name, t_idx):
    """
    Computes Relative L1, L2, and Linf norms of the difference (B - A).
    Rel Error = ||B - A|| / ||A||
    """
    try:
        with xr.open_dataset(file_a) as ds_a, xr.open_dataset(file_b) as ds_b:
            # Check Variable
            if var_name not in ds_a or var_name not in ds_b:
                print(f"  [Skip] Var '{var_name}' missing in {file_a.name} or {file_b.name}")
                return None

            da_a = ds_a[var_name]
            da_b = ds_b[var_name]

            # Time Selection
            if 'time' in da_a.dims:
                # Handle negative index safely
                t_sel_a = t_idx if t_idx >= 0 else da_a.sizes['time'] + t_idx
                t_sel_b = t_idx if t_idx >= 0 else da_b.sizes['time'] + t_idx

                # Clip to valid range
                t_sel_a = np.clip(t_sel_a, 0, da_a.sizes['time'] - 1)
                t_sel_b = np.clip(t_sel_b, 0, da_b.sizes['time'] - 1)

                val_a = da_a.isel(time=int(t_sel_a)).values
                val_b = da_b.isel(time=int(t_sel_b)).values
            else:
                val_a = da_a.values
                val_b = da_b.values

            # Check Shapes
            if val_a.shape != val_b.shape:
                print(f"  [Skip] Shape mismatch: {val_a.shape} vs {val_b.shape}")
                return None

            # --- Calculation ---
            # Reference Norms (Sim A)
            flat_a = val_a.flatten()
            norm_l1_ref = np.mean(np.abs(flat_a))
            norm_l2_ref = np.sqrt(np.mean(flat_a ** 2))
            norm_linf_ref = np.max(np.abs(flat_a))

            # Avoid div by zero
            if norm_l2_ref < 1e-15:
                print(f"  [Skip] Reference field is effectively zero for {file_a.name}")
                return None

            # Difference Field
            diff = val_b - val_a
            flat_diff = diff.flatten()
            abs_diff = np.abs(flat_diff)

            # Relative Errors
            rel_l1 = np.mean(abs_diff) / norm_l1_ref
            rel_l2 = np.sqrt(np.mean(flat_diff ** 2)) / norm_l2_ref
            rel_linf = np.max(abs_diff) / norm_linf_ref

            return rel_l1, rel_l2, rel_linf

    except Exception as e:
        print(f"  [Error] Processing pair failed: {e}")
        return None


# ==========================================
# 3. MAIN EXECUTION
# ==========================================

def main():
    path_a = Path(FOLDER_A).expanduser().resolve()
    path_b = Path(FOLDER_B).expanduser().resolve()

    if not path_a.exists() or not path_b.exists():
        print("Error: One or both data folders do not exist.")
        return

    # 1. Map Files by Parameter
    files_a_map = {}
    for f in path_a.glob("*.nc"):
        p = get_parameter_from_filename(f.name)
        if p is not None: files_a_map[p] = f

    files_b_map = {}
    for f in path_b.glob("*.nc"):
        p = get_parameter_from_filename(f.name)
        if p is not None: files_b_map[p] = f

    # 2. Find common parameters
    common_params = sorted(list(set(files_a_map.keys()) & set(files_b_map.keys())))

    if not common_params:
        print("No matching files found based on parameter regex.")
        return

    print(f"Found {len(common_params)} matching pairs. computing errors...")

    # 3. Compute Loops
    results = []
    for p in common_params:
        f_a = files_a_map[p]
        f_b = files_b_map[p]

        metrics = compute_difference_metrics(f_a, f_b, VAR_NAME, TIME_INDEX)

        if metrics:
            l1, l2, linf = metrics
            results.append({'param': p, 'l1': l1, 'l2': l2, 'linf': linf})
            print(f"  [OK] Param={p}: Rel L2={l2:.2e}")

    if not results:
        print("No valid error metrics computed.")
        return

    # 4. Prepare Plot Data
    results.sort(key=lambda x: x['param'])
    x_vals = [r['param'] for r in results]
    l1_vals = [r['l1'] for r in results]
    l2_vals = [r['l2'] for r in results]
    linf_vals = [r['linf'] for r in results]

    # 5. Plotting
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    # Colors (Colorblind safe sequence)
    cmap = plt.get_cmap('cividis')
    colors = cmap(np.linspace(0, 0.9, 3))

    ax.plot(x_vals, l1_vals, marker='o', linestyle='-', color=colors[0], label=r'Relative $L_1$')
    ax.plot(x_vals, l2_vals, marker='s', linestyle='--', color=colors[1], label=r'Relative $L_2$')
    ax.plot(x_vals, linf_vals, marker='^', linestyle=':', color=colors[2], label=r'Relative $L_\infty$')

    # Formatting
    ax.set_title(f"Difference Evolution: {VAR_LABEL}", fontweight='bold')
    ax.set_xlabel(PARAM_LABEL)
    ax.set_ylabel(r"Relative Error $\left(\frac{||\phi_B - \phi_A||}{||\phi_A||}\right)$")

    # Log Scales
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Fancy Ticks for Log Scale
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(labelOnlyBase=False))

    # Grid
    ax.grid(True, which="major", ls="-", alpha=0.5)
    ax.grid(True, which="minor", ls=":", alpha=0.3)

    ax.legend()

    save_name = "relative_error_evolution.png"
    print(f"Saving plot to {save_name}...")
    plt.savefig(save_name, dpi=300)
    plt.show()


if __name__ == "__main__":
    main()