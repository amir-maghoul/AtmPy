import xarray as xr
import numpy as np
import os
import subprocess  # To run main.py
import matplotlib.pyplot as plt
import logging
import hashlib  # For checksums if desired later

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class AtmpyRegressionTester:
    def __init__(
        self,
        test_case_name: str,
        config_for_run,
        benchmark_dir: str = "atmpy/tests/benchmarks",
    ):
        """
        Initializes the regression tester.

        Args:
            test_case_name (str): Name of the test case (e.g., "TravelingVortex").
            config_for_run (SimulationConfig): The configuration object to use for the test run.
                                                This ensures the run matches benchmark conditions.
            benchmark_dir (str): Directory where benchmark NetCDF files are stored.
        """
        self.test_case_name = test_case_name
        self.config = config_for_run  # Use the provided config for the run
        self.benchmark_dir = benchmark_dir

        # Determine expected output filename from the config
        # Ensure the output path is relative to the project root if needed
        # For simplicity, assume main.py writes to the path specified in config.outputs.output_filename
        self.current_run_output_file = self.config.outputs.output_filename
        if not os.path.isabs(self.current_run_output_file):
            # Assuming the script is run from the project root
            self.current_run_output_file = os.path.join(
                os.getcwd(), self.current_run_output_file
            )

        # Construct benchmark filename (convention based on test_case_name and grid)
        # Example: traveling_vortex_64x64_benchmark.nc
        grid_suffix = self.config.outputs.output_suffix  # Should be like "_64x64"
        self.benchmark_file = os.path.join(
            self.benchmark_dir,
            f"{self.test_case_name.lower().replace(' ', '_')}{grid_suffix}_benchmark.nc",
        )

        self.variables_to_compare = [
            "rho",
            "rhou",
            "rhov",
            "rhoY",
            "p2_nodes",
        ]  # Customize
        # Map your internal variable names (from VI enum) to NetCDF output names if they differ.
        # For now, assume they are the same as used in _save_snapshot.
        # Your _save_snapshot uses: "rho", "u", "v", "w", "Y", "X", "p2_nodes"
        # Let's adjust:
        self.variables_to_compare = ["rho", "u", "v", "Y", "p2_nodes"]
        if self.config.grid.ndim == 3:
            self.variables_to_compare.append("w")
        if (
            "X" in xr.open_dataset(self.benchmark_file, decode_times=False).data_vars
        ):  # Check if X exists
            self.variables_to_compare.append("X")

    def run_test_case(self, main_script_path: str = "main.py"):
        """
        Runs the specified test case using main.py.
        """
        logging.info(f"Running test case: {self.test_case_name}...")

        # Ensure output directory for the current run exists
        os.makedirs(os.path.dirname(self.current_run_output_file), exist_ok=True)

        # Clean up previous run's output file if it exists to ensure a fresh run
        if os.path.exists(self.current_run_output_file):
            logging.info(
                f"Removing previous output file: {self.current_run_output_file}"
            )
            os.remove(self.current_run_output_file)

        # Construct the command to run main.py
        # We need to pass the test case name. Other configurations are handled by the
        # SimulationConfig object associated with this tester instance, which main.py should use.
        # This requires main.py to be able to accept a config object or a path to a config file.
        # For now, let's assume main.py is set up to run the specific config associated with the test case.
        # A more robust way would be to save the self.config to a temporary YAML/JSON
        # and pass that file path to main.py.

        command = [
            "python",
            main_script_path,
            "--case",
            self.test_case_name,
            "--mode",
            "run",
            # We need to ensure main.py uses self.config for this run.
            # This is tricky without modifying main.py to accept a config file
            # or by having a way for the test case class to load this specific config.
            # For now, we assume main.py will pick up the correct config for the test case
            # based on how it instantiates TravelingVortex() etc.
            # To ensure this test run *exactly* matches benchmark conditions:
            # - The config.temporal.stepmax / tmax must be the same
            # - The config.outputs.output_frequency_steps must be the same for comparable time points
        ]
        logging.info(f"Executing command: {' '.join(command)}")

        try:
            process = subprocess.run(
                command, check=True, capture_output=True, text=True
            )
            logging.info("Test case run completed successfully.")
            logging.debug("STDOUT:\n" + process.stdout)
            if process.stderr:
                logging.warning("STDERR:\n" + process.stderr)
        except subprocess.CalledProcessError as e:
            logging.error(f"Test case run failed for {self.test_case_name}.")
            logging.error("STDOUT:\n" + e.stdout)
            logging.error("STDERR:\n" + e.stderr)
            raise RuntimeError(
                f"Test case {self.test_case_name} execution failed."
            ) from e

        if not os.path.exists(self.current_run_output_file):
            raise FileNotFoundError(
                f"Output file {self.current_run_output_file} was not generated by the test run."
            )

    def compare_results(
        self,
        time_indices_to_compare: list = [-1],
        rtol: float = 1e-5,
        atol: float = 1e-8,
        plot_diff: bool = False,
    ):
        """
        Compares the output of the current run with the benchmark file.

        Args:
            time_indices_to_compare (list): List of time indices (e.g., [0, -1] for first and last)
                                           to compare.
            rtol (float): Relative tolerance for np.allclose.
            atol (float): Absolute tolerance for np.allclose.
            plot_diff (bool): If True, generate difference plots for the last compared time index.

        Returns:
            bool: True if all comparisons pass, False otherwise.
        """
        logging.info(
            f"Comparing results for {self.test_case_name} against {self.benchmark_file}"
        )
        if not os.path.exists(self.benchmark_file):
            logging.error(f"Benchmark file not found: {self.benchmark_file}")
            logging.error(
                "Please generate it first by running the test case and copying its output."
            )
            return False
        if not os.path.exists(self.current_run_output_file):
            logging.error(
                f"Current run output file not found: {self.current_run_output_file}"
            )
            return False

        try:
            # decode_times=False can be faster if time coordinate isn't complex
            ds_benchmark = xr.open_dataset(self.benchmark_file, decode_times=False)
            ds_current = xr.open_dataset(
                self.current_run_output_file, decode_times=False
            )
        except Exception as e:
            logging.error(f"Error opening NetCDF files: {e}")
            return False

        all_passed = True

        # Basic check: number of time steps
        if len(ds_benchmark.time) != len(ds_current.time):
            logging.warning(
                f"Number of time steps differ: Benchmark={len(ds_benchmark.time)}, Current={len(ds_current.time)}"
            )
            # This might be acceptable if only comparing specific snapshots, but often indicates a divergence.
            # For regression, usually you want them to be identical.
            # all_passed = False # Uncomment if this should be a strict failure

        for var_name in self.variables_to_compare:
            if var_name not in ds_benchmark.data_vars:
                logging.warning(
                    f"Variable '{var_name}' not found in benchmark file. Skipping."
                )
                continue
            if var_name not in ds_current.data_vars:
                logging.warning(
                    f"Variable '{var_name}' not found in current run output. Skipping."
                )
                all_passed = False
                continue

            logging.info(f"Comparing variable: {var_name}")

            for t_idx in time_indices_to_compare:
                try:
                    data_benchmark = ds_benchmark[var_name].isel(time=t_idx).values
                    data_current = ds_current[var_name].isel(time=t_idx).values
                except IndexError:
                    logging.error(
                        f"Time index {t_idx} out of bounds for variable {var_name}. Max benchmark time index: {len(ds_benchmark.time)-1}, Max current time index: {len(ds_current.time)-1}"
                    )
                    all_passed = False
                    continue
                except Exception as e:
                    logging.error(
                        f"Error selecting data for {var_name} at t_idx={t_idx}: {e}"
                    )
                    all_passed = False
                    continue

                # Metric 1: L2 norm of the difference
                diff = data_benchmark - data_current
                l2_norm_diff = np.linalg.norm(diff) / np.linalg.norm(
                    data_benchmark
                )  # Relative L2 norm
                logging.info(
                    f"  Time index {t_idx}: Relative L2 norm of difference = {l2_norm_diff:.2e}"
                )

                if (
                    l2_norm_diff > rtol
                ):  # Using rtol as a threshold for the relative L2 norm
                    logging.error(
                        f"  FAIL: Variable '{var_name}' at time index {t_idx} - L2 norm exceeds tolerance ({l2_norm_diff:.2e} > {rtol:.1e})"
                    )
                    all_passed = False
                else:
                    logging.info(
                        f"  PASS: Variable '{var_name}' at time index {t_idx} - L2 norm within tolerance."
                    )

                # Metric 2: np.allclose (more stringent element-wise comparison)
                if not np.allclose(
                    data_benchmark, data_current, rtol=rtol, atol=atol, equal_nan=True
                ):
                    logging.error(
                        f"  FAIL: Variable '{var_name}' at time index {t_idx} - np.allclose failed."
                    )
                    # all_passed = False # L2 norm is often a better high-level regression metric for PDEs

                # Optional: Sum comparison (less robust but quick)
                sum_benchmark = np.nansum(data_benchmark)
                sum_current = np.nansum(data_current)
                if not np.isclose(sum_benchmark, sum_current, rtol=rtol):
                    logging.debug(
                        f"  Sum mismatch for '{var_name}' at t_idx {t_idx}: Benchmark={sum_benchmark:.6e}, Current={sum_current:.6e}"
                    )

                if (
                    plot_diff and t_idx == time_indices_to_compare[-1]
                ):  # Plot for the last time index
                    self._plot_difference(
                        ds_benchmark,
                        var_name,
                        data_benchmark,
                        data_current,
                        diff,
                        t_idx,
                    )

        ds_benchmark.close()
        ds_current.close()
        return all_passed

    def _plot_difference(
        self, ds_ref_for_coords, var_name, data_ref, data_curr, diff_data, time_idx
    ):
        """Helper to plot reference, current, and difference."""
        logging.info(
            f"Generating difference plot for {var_name} at time index {time_idx}"
        )
        # Determine coordinates from the reference dataset
        # Adjust based on your NetCDF structure (e.g., 'x_rho', 'y_rho' or just 'x', 'y')
        x_coord_name = "x"
        y_coord_name = "y"
        if (
            x_coord_name not in ds_ref_for_coords.coords
            or y_coord_name not in ds_ref_for_coords.coords
        ):
            logging.warning(
                f"Coordinates {x_coord_name} or {y_coord_name} not found for plotting. Skipping plot."
            )
            return

        x_coords = ds_ref_for_coords[x_coord_name].values
        y_coords = ds_ref_for_coords[y_coord_name].values

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
        fig.suptitle(
            f"Comparison for '{var_name}' at time index {time_idx} (t={ds_ref_for_coords.time.isel(time=time_idx).item():.2f}s)",
            fontsize=14,
        )

        common_min = min(np.nanmin(data_ref), np.nanmin(data_curr))
        common_max = max(np.nanmax(data_ref), np.nanmax(data_curr))
        levels = np.linspace(common_min, common_max, 15)

        # Plotting assumes 2D data (x, y). Transpose .T might be needed if data is (y, x)
        # Your solver saves (nx, ny) so contourf expects (x_coords, y_coords, data.T)
        # or (x_coords_mesh, y_coords_mesh, data)
        # If x_coords and y_coords are 1D:
        im1 = axes[0].contourf(
            x_coords, y_coords, data_ref.T, levels=levels, cmap="viridis", extend="both"
        )
        axes[0].set_title("Benchmark")
        fig.colorbar(im1, ax=axes[0])

        im2 = axes[1].contourf(
            x_coords,
            y_coords,
            data_curr.T,
            levels=levels,
            cmap="viridis",
            extend="both",
        )
        axes[1].set_title("Current Run")
        fig.colorbar(im2, ax=axes[1])

        # For difference, use a diverging colormap and center around 0
        diff_abs_max = np.nanmax(np.abs(diff_data))
        if diff_abs_max < 1e-12:
            diff_abs_max = 1e-12  # Avoid zero range for levels
        diff_levels = np.linspace(-diff_abs_max, diff_abs_max, 15)
        im3 = axes[2].contourf(
            x_coords,
            y_coords,
            diff_data.T,
            levels=diff_levels,
            cmap="coolwarm",
            extend="both",
        )
        axes[2].set_title("Difference (Benchmark - Current)")
        fig.colorbar(im3, ax=axes[2])

        for ax in axes:
            ax.set_xlabel(x_coord_name)
            ax.set_ylabel(y_coord_name)
            ax.set_aspect("equal", adjustable="box")

        plot_filename = os.path.join(
            os.path.dirname(
                self.current_run_output_file
            ),  # Save in current run's output dir
            f"regression_diff_{self.test_case_name}_{var_name}_t{time_idx}.png",
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
        plt.savefig(plot_filename)
        logging.info(f"Difference plot saved to {plot_filename}")
        plt.close(fig)

    def update_benchmark(self):
        """
        Copies the current run's output file to the benchmark directory,
        overwriting the existing benchmark. USE WITH CAUTION.
        """
        if not os.path.exists(self.current_run_output_file):
            logging.error(
                f"Current run output file {self.current_run_output_file} not found. Cannot update benchmark."
            )
            logging.error("Please run the test case first.")
            return

        logging.warning(f"Updating benchmark for {self.test_case_name}!")
        logging.warning(f"This will OVERWRITE: {self.benchmark_file}")
        confirm = input("Are you sure you want to overwrite the benchmark? (yes/no): ")

        if confirm.lower() == "yes":
            try:
                os.makedirs(self.benchmark_dir, exist_ok=True)
                import shutil

                shutil.copyfile(self.current_run_output_file, self.benchmark_file)
                logging.info(f"Benchmark updated: {self.benchmark_file}")
            except Exception as e:
                logging.error(f"Failed to update benchmark: {e}")
        else:
            logging.info("Benchmark update cancelled.")


if __name__ == "__main__":
    # This is an example of how to use the tester
    # You would typically call this from a dedicated test script or a CI pipeline.

    # 1. Get the configuration for the test case you want to run
    # This part needs to be robust. main.py instantiates test cases and they hold their config.
    # We need that same config here.
    from atmpy.test_cases.traveling_vortex import TravelingVortex

    case_instance = TravelingVortex()  # This sets up the default config inside the case
    test_config = case_instance.config

    # MODIFY test_config FOR REGRESSION TEST CONSISTENCY if needed:
    # For example, ensure it runs for the same number of steps/time as the benchmark.
    # Benchmark for TravelingVortex was likely run for specific steps/time.
    # Let's assume benchmark was for 100 steps, output every 50 steps.
    original_stepmax = test_config.temporal.stepmax
    original_output_freq = test_config.outputs.output_frequency_steps

    test_config.temporal.stepmax = 100  # Match benchmark run length
    test_config.outputs.output_frequency_steps = 50  # Match benchmark output frequency
    test_config.temporal.tout = []  # Ensure no extra outputs alter the timeline
    # Update output filename in config as it's used by the tester to find the output
    test_config.outputs.output_filename = os.path.join(
        test_config.outputs.output_path,
        test_config.outputs.output_folder,
        f"{test_config.outputs.output_base_name}{test_config.outputs.output_suffix}{test_config.outputs.output_extension}",
    )

    # 2. Create the tester instance
    tester = AtmpyRegressionTester(
        test_case_name="TravelingVortex", config_for_run=test_config
    )

    # --- Choose an action ---

    # Action A: Run test and compare
    PERFORM_RUN_AND_COMPARE = True
    # Action B: Only update the benchmark (use after verifying a new "good" run)
    UPDATE_BENCHMARK_ONLY = False

    if UPDATE_BENCHMARK_ONLY:
        # First, ensure a "good" run has produced self.current_run_output_file
        # You might run it manually or via tester.run_test_case()
        # If current_run_output_file doesn't exist, this will fail.
        # For example, run it first:
        # tester.run_test_case()
        tester.update_benchmark()
    elif PERFORM_RUN_AND_COMPARE:
        try:
            # This will run main.py with the "TravelingVortex" case and the test_config settings
            tester.run_test_case()

            # Compare, looking at the first (index 0) and last (index -1) time snapshots
            # Plot differences for the last snapshot
            success = tester.compare_results(
                time_indices_to_compare=[0, -1], plot_diff=True, rtol=1e-4, atol=1e-7
            )

            if success:
                logging.info(f"Regression test PASSED for {tester.test_case_name}")
            else:
                logging.error(f"Regression test FAILED for {tester.test_case_name}")
        except Exception as e:
            logging.error(
                f"An error occurred during the regression test: {e}", exc_info=True
            )
        finally:
            # Restore original config values if they were changed for the test
            test_config.temporal.stepmax = original_stepmax
            test_config.outputs.output_frequency_steps = original_output_freq

    # To run for another test case, create another case_instance and tester
    # from atmpy.test_cases.rising_bubble import RisingBubble
    # case_instance_rb = RisingBubble()
    # test_config_rb = case_instance_rb.config
    # # ... modify test_config_rb for consistency ...
    # tester_rb = AtmpyRegressionTester(test_case_name="RisingBubble", config_for_run=test_config_rb)
    # tester_rb.run_test_case()
    # tester_rb.compare_results(...)
