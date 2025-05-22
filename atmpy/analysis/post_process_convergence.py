import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s"
)

BASE_OUTPUT_DIR = "outputs"
MANIFEST_FILE = os.path.join(BASE_OUTPUT_DIR, "output_files_manifest.txt")
VARIABLES_TO_ANALYZE = ["rho", "u", "v", "Y"]  # Should match run_convergence.py


def calculate_solution_change_norms(initial_array, final_array):
    """Calculates L1 (mean abs), L2 (RMS), and L-infinity norms of (final_array - initial_array)."""
    if initial_array.shape != final_array.shape:
        raise ValueError(
            f"Initial and final array shapes must match. Got {initial_array.shape} and {final_array.shape}"
        )
    diff = final_array - initial_array
    l1_norm_mean = np.mean(np.abs(diff))
    l2_norm_rms = np.sqrt(np.mean(diff**2))  # RMS of the change
    linf_norm = np.max(np.abs(diff))  # Max absolute change
    return l1_norm_mean, l2_norm_rms, linf_norm


def main():
    logging.info(
        "Starting post-processing for convergence study: Norm of (Final - Initial)."
    )

    if not os.path.exists(MANIFEST_FILE):
        logging.error(f"Manifest file not found: {MANIFEST_FILE}")
        return

    output_files_data = {}
    with open(MANIFEST_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            nx_str, filepath = line.split(":", 1)
            try:
                output_files_data[int(nx_str)] = filepath
            except ValueError:
                logging.warning(f"Could not parse Nx value from line: {line}")
                continue

    if not output_files_data:
        logging.error("No valid data found in manifest file.")
        return

    sorted_nxs = sorted(output_files_data.keys())
    logging.info(f"Found data for Nx values: {sorted_nxs}")

    # Store norms of (Final - Initial) for each Nx and each variable
    # Structure: {var_name: {"nx": [], "l1_mean": [], "l2_rms": [], "linf": []}}
    all_solution_change_norms = {
        var: {"nx": [], "l1_mean": [], "l2_rms": [], "linf": []}
        for var in VARIABLES_TO_ANALYZE
    }

    # --- Step 1: Load data and calculate Norm(Final - Initial) for each resolution ---
    for nx_val in sorted_nxs:
        filepath = output_files_data[nx_val]
        logging.info(f"\nProcessing Nx = {nx_val} from file: {filepath}")

        try:
            ds = xr.open_dataset(filepath)
        except FileNotFoundError:
            logging.error(f"File not found: {filepath}. Skipping Nx={nx_val}.")
            continue
        except Exception as e:
            logging.error(f"Error opening {filepath}: {e}. Skipping Nx={nx_val}.")
            continue

        if "time" not in ds.coords or len(ds.time) < 2:
            logging.error(
                f"Dataset for Nx={nx_val} (file: {filepath}) does not contain at least two time steps. "
                f"Found {len(ds.time) if 'time' in ds.coords else 0} time steps. Skipping."
            )
            ds.close()
            continue

        logging.info(f"  Time values in {filepath}: {ds.time.values}")

        # Flag to track if this Nx run is valid for all variables
        current_nx_valid = True

        for var_name in VARIABLES_TO_ANALYZE:
            if var_name not in ds:
                logging.warning(
                    f"Variable '{var_name}' not found in dataset for Nx={nx_val} (file: {filepath}). Skipping this variable for this Nx."
                )
                current_nx_valid = (
                    False  # Mark this Nx as potentially problematic for this variable
                )
                continue

            try:
                initial_data = ds[var_name].isel(time=0).data
                final_data = ds[var_name].isel(time=-1).data

                if np.allclose(
                    initial_data, final_data, atol=1e-12, rtol=1e-9
                ):  # Increased tolerance slightly for very small changes
                    logging.warning(
                        f"  WARNING: For Nx={nx_val}, Var='{var_name}', final data is essentially identical to initial data!"
                    )
                    # Depending on the problem, this might be expected for some T_FINAL or might indicate a stalled simulation.
                    # For a traveling vortex, this is unexpected if T_FINAL > 0 and velocity > 0.

                l1_norm, l2_rms_norm, linf_norm = calculate_solution_change_norms(
                    initial_data, final_data
                )
                logging.info(
                    f"  Nx={nx_val}, Var='{var_name}': Norms(Final-Initial): L1_mean={l1_norm:.4e}, L2_RMS={l2_rms_norm:.4e}, L_inf={linf_norm:.4e}"
                )

                all_solution_change_norms[var_name]["nx"].append(nx_val)
                all_solution_change_norms[var_name]["l1_mean"].append(l1_norm)
                all_solution_change_norms[var_name]["l2_rms"].append(l2_rms_norm)
                all_solution_change_norms[var_name]["linf"].append(linf_norm)

            except Exception as e:
                logging.error(
                    f"  Error processing variable '{var_name}' for Nx={nx_val}: {e}"
                )
                current_nx_valid = False
                # Remove any partially added data for this nx_val for this var if an error occurred mid-variable processing
                if (
                    all_solution_change_norms[var_name]["nx"]
                    and all_solution_change_norms[var_name]["nx"][-1] == nx_val
                ):
                    all_solution_change_norms[var_name]["nx"].pop()
                    all_solution_change_norms[var_name]["l1_mean"].pop()
                    all_solution_change_norms[var_name]["l2_rms"].pop()
                    all_solution_change_norms[var_name]["linf"].pop()
                break  # Stop processing other variables for this Nx if one fails critically

        ds.close()

    # --- Step 2: Plotting Norm(Final - Initial) vs Nx ---
    logging.info("\n--- Plotting Norm of (Final - Initial) vs Nx ---")

    for var_name in VARIABLES_TO_ANALYZE:
        if not all_solution_change_norms[var_name]["nx"]:
            logging.info(f"No data to plot for variable {var_name}")
            continue

        plot_nxs = np.array(all_solution_change_norms[var_name]["nx"])
        l1_errors = np.array(all_solution_change_norms[var_name]["l1_mean"])
        l2_rms_errors = np.array(all_solution_change_norms[var_name]["l2_rms"])
        linf_errors = np.array(all_solution_change_norms[var_name]["linf"])

        # Sort by Nx just in case they weren't processed in strict order (though sorted_nxs should handle this)
        sort_indices = np.argsort(plot_nxs)
        plot_nxs = plot_nxs[sort_indices]
        l1_errors = l1_errors[sort_indices]
        l2_rms_errors = l2_rms_errors[sort_indices]
        linf_errors = linf_errors[sort_indices]

        plt.figure(figsize=(10, 7))

        if len(plot_nxs) > 0:
            plt.plot(
                plot_nxs,
                l1_errors,
                marker="s",
                linestyle="--",
                label=f"L1_mean (Final-Initial)",
            )
            plt.plot(
                plot_nxs,
                l2_rms_errors,
                marker="o",
                linestyle="-",
                label=f"L2_RMS (Final-Initial)",
            )
            plt.plot(
                plot_nxs,
                linf_errors,
                marker="^",
                linestyle=":",
                label=f"L_inf (Final-Initial)",
            )

        plt.xlabel("Nx (Grid Resolution)")
        plt.ylabel("Norm of (Final State - Initial State)")
        plt.title(
            f"Evolution Magnitude for {var_name} vs. Grid Resolution (T_final={ds.time.values[-1]}s)"
        )  # Assumes last ds is available for T_FINAL

        plt.xscale("log", base=2)  # Good for Nx like 16, 32, 64, 128
        plt.yscale("log")  # Norms often span orders of magnitude

        plt.xticks(
            plot_nxs, labels=[str(n) for n in plot_nxs]
        )  # Ensure all Nx values are ticked
        plt.legend()
        plt.grid(True, which="both", ls="-")

        plot_filename = os.path.join(
            BASE_OUTPUT_DIR, f"solution_change_norm_vs_nx_{var_name}.png"
        )
        plt.savefig(plot_filename)
        logging.info(f"Saved solution change norm plot to {plot_filename}")
        plt.close()

    # --- Step 3: (Optional) Calculate and Print Order of Convergence of the Norms themselves ---
    # This tells us how rapidly the measured "total change" converges as resolution increases.
    # This is different from the order of accuracy of the scheme for the solution itself.
    logging.info("\n--- Convergence Orders of the Norm(Final - Initial) ---")
    refinement_factor = 2.0  # Assuming grid doubles each time

    for var_name in VARIABLES_TO_ANALYZE:
        logging.info(f"\nVariable: {var_name}")
        for norm_key in ["l1_mean", "l2_rms", "linf"]:
            norms_values = np.array(all_solution_change_norms[var_name][norm_key])
            current_nxs = np.array(all_solution_change_norms[var_name]["nx"])

            if len(norms_values) > 1:
                logging.info(f"  {norm_key.upper()} of (Final-Initial):")
                # Example: if norms are [N32, N64, N128], we compare (N32, N64) then (N64, N128) etc.
                # Then estimate order based on |N_coarse - N_ref| / |N_fine - N_ref| if N_ref is known (e.g. highest res)
                # Or, if we assume N(h) = N_exact + C*h^p, then N(h) - N(h/2) approaches C*h^p
                # (N(h) - N(h/2)) / (N(h/2) - N(h/4)) should be refinement_factor^p

                # Compare differences between successive norms
                # e.g., |norm_at_Nx64 - norm_at_Nx32| vs |norm_at_Nx128 - norm_at_Nx64|
                # This is getting complicated again. Let's just print the sequence.
                # If they are converging, the difference between successive norms should decrease.
                # for j in range(len(norms_values) -1):
                # logging.info(f"    Nx={current_nxs[j]} -> Nx={current_nxs[j+1]}: {norms_values[j]:.4e} -> {norms_values[j+1]:.4e} (Diff: {abs(norms_values[j+1]-norms_values[j]):.4e})")

                # A simpler approach for order if norms are converging to N_exact:
                # error_coarse = abs(norms_values[j] - norms_values[-1]) # Diff from highest res norm
                # error_fine = abs(norms_values[j+1] - norms_values[-1]) # Diff from highest res norm
                # Requires at least 3 resolutions to compute one order value this way.
                if len(norms_values) >= 3:
                    logging.info(
                        f"  Order estimation for {norm_key.upper()} (using 3 finest resolutions as {sorted_nxs[-3:]}):"
                    )
                    # e.g., nxs are 32, 64, 128, 256. norms_values[-3:] are for 64, 128, 256
                    # N_c = norm(64), N_m = norm(128), N_f = norm(256)
                    # Order p = log( (N_c - N_m) / (N_m - N_f) ) / log(refinement_factor)
                    # This assumes N(h) = N_exact + C*h^p + O(h^{p+1})
                    # (N(2h) - N(h)) / (N(h) - N(h/2)) = ( (C(2h)^p - Ch^p) / (Ch^p - C(h/2)^p) ) ...
                    # = (2^p - 1) / (1 - (1/2)^p) = 2^p
                    # So order p = log(ratio) / log(2)

                    # Need to ensure the differences are consistently signed or use abs,
                    # but for convergence, N(h) - N_exact should have the same sign.
                    # Let's use differences from the finest available solution as an estimate of error.
                    n_ref = norms_values[-1]  # Norm from the finest grid as reference
                    for j in range(
                        len(norms_values) - 2
                    ):  # Iterate up to the third to last
                        # Error = |N(h) - N_ref|
                        err_coarse = abs(norms_values[j] - n_ref)
                        err_fine = abs(norms_values[j + 1] - n_ref)

                        nx_c = current_nxs[j]
                        nx_f = current_nxs[j + 1]

                        if (
                            err_coarse > 1e-15
                            and err_fine > 1e-15
                            and err_coarse > err_fine
                        ):  # Basic check for convergence
                            order = np.log(err_coarse / err_fine) / np.log(
                                nx_f / nx_c
                            )  # nx_f/nx_c is refinement ratio
                            logging.info(
                                f"    Order between Nx={nx_c} and Nx={nx_f} (ref. Nx={current_nxs[-1]}): {order:.2f} "
                                f"(errors: {err_coarse:.2e}, {err_fine:.2e})"
                            )
                        else:
                            logging.info(
                                f"    Cannot compute order or non-convergent pattern for {norm_key.upper()} "
                                f"between Nx={nx_c} and Nx={nx_f} (ref. Nx={current_nxs[-1]}) "
                                f"(errors: {err_coarse:.2e}, {err_fine:.2e})"
                            )
                else:
                    logging.info(
                        f"  Not enough resolution pairs for {norm_key.upper()} to estimate order robustly."
                    )

            elif not list(norms_values):  # Check if the list is empty
                logging.info(
                    f"  No {norm_key.upper()} norm data to compute convergence for {var_name}."
                )

    logging.info("Post-processing complete.")


if __name__ == "__main__":
    main()
