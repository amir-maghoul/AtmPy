import os
import subprocess
import shutil
import logging
import time
import pickle  # Add import
import tempfile  # Add import

from atmpy.test_cases.traveling_vortex import TravelingVortex

# from atmpy.configuration.simulation_configuration import SimulationConfig # Keep if type hinting

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s"
)

TEST_CASE_NAME = "TravelingVortex"
BASE_OUTPUT_DIR = "outputs"
MAIN_SCRIPT_PATH = (
    "/home/amir/Projects/Python/Atmpy/main.py"  # Path to your main execution script
)
RESOLUTIONS_NX = [32, 64, 128, 256]
T_FINAL = 0.01
BASE_DT_FOR_COARSEST = 0.005
EXPECTED_VARIABLES = ["rho", "u", "v", "Y"]


def run_single_atmpy_case(
    config_obj_for_info, case_name, main_script, config_pickle_path: str
):  # Add pickle path
    """
    Runs a single atmpy simulation using main.py.
    config_obj_for_info is used for logging info here, the actual config is passed via pickle.
    """
    # Ensure output directory for this specific run exists (based on info from config_obj_for_info)
    # The actual main.py run will use the filename from the pickled config.
    # It's good if they match. The pickled config_obj_for_info.outputs.output_filename should be correct.
    os.makedirs(
        os.path.dirname(config_obj_for_info.outputs.output_filename), exist_ok=True
    )
    if os.path.exists(config_obj_for_info.outputs.output_filename):
        logging.info(
            f"Removing previous output (if any): {config_obj_for_info.outputs.output_filename}"
        )
        os.remove(config_obj_for_info.outputs.output_filename)

    command = [
        "python",
        main_script,
        "--case",
        case_name,
        "--mode",
        "run",
        "--config-pickle-path",
        config_pickle_path,  # Pass the path to the pickled config
    ]
    logging.info(f"Executing: {' '.join(command)} for Nx={config_obj_for_info.grid.nx}")
    logging.info(
        f"Expected output file (from convergence script config): {config_obj_for_info.outputs.output_filename}"
    )
    logging.info(
        f"Using dt: {config_obj_for_info.temporal.dtfixed}, stepmax: {config_obj_for_info.temporal.stepmax}"
    )

    try:
        process = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=600
        )
        logging.info(f"Run for Nx={config_obj_for_info.grid.nx} completed.")
        if process.stdout:
            logging.debug(
                f"STDOUT (Nx={config_obj_for_info.grid.nx}):\n{process.stdout[-500:]}"
            )
        if process.stderr:
            logging.warning(
                f"STDERR (Nx={config_obj_for_info.grid.nx}):\n{process.stderr}"
            )
        return (
            config_obj_for_info.outputs.output_filename
        )  # Return the filename as known by this script
    except subprocess.TimeoutExpired:
        logging.error(f"Run for Nx={config_obj_for_info.grid.nx} TIMED OUT.")
        return None
    except subprocess.CalledProcessError as e:
        logging.error(f"Run for Nx={config_obj_for_info.grid.nx} FAILED.")
        logging.error("STDOUT:\n" + e.stdout)
        logging.error("STDERR:\n" + e.stderr)
        return None


def main():
    logging.info("Starting convergence study for: " + TEST_CASE_NAME)
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    output_files = {}
    coarsest_nx = RESOLUTIONS_NX[0]

    for i, nx_current in enumerate(RESOLUTIONS_NX):
        logging.info(f"\n--- Preparing run for Nx = {nx_current} ---")

        # Instantiate the case. Its __init__ will call setup() if no override is given.
        # This gives us a default config populated by the case's setup() method.
        current_case_instance = (
            TravelingVortex()
        )  # Or dynamically based on TEST_CASE_NAME
        config = current_case_instance.config  # This is the config to be modified

        # Modify the config for the current resolution and settings
        grid_updates = {"nx": nx_current, "ny": nx_current}
        current_case_instance.set_grid_configuration(
            grid_updates
        )  # This updates config.grid
        current_case_instance._update_output_suffix()  # This updates config.outputs.output_suffix

        if i == 0:
            config.temporal.dtfixed = BASE_DT_FOR_COARSEST
        else:
            ref_dt = BASE_DT_FOR_COARSEST
            ref_nx = coarsest_nx
            config.temporal.dtfixed = ref_dt * (ref_nx / nx_current)

        if config.temporal.dtfixed <= 0:
            logging.error(
                f"Calculated dt is not positive ({config.temporal.dtfixed}). Aborting."
            )
            return
        config.temporal.stepmax = int(T_FINAL / config.temporal.dtfixed) + 1
        config.outputs.output_frequency_steps = config.temporal.stepmax
        config.temporal.tout = [0.0, T_FINAL]
        # Set dtfixed0 if your solver uses it separately, e.g., for initial projection
        config.temporal.dtfixed0 = config.temporal.dtfixed
        config.outputs.output_frequency_steps = config.temporal.stepmax

        run_specific_output_folder_name = f"run_nx{nx_current}"
        run_specific_output_folder_path = os.path.join(
            BASE_OUTPUT_DIR, run_specific_output_folder_name
        )

        config.outputs.output_path = "."  # Output relative to where main.py is run
        config.outputs.output_folder = (
            run_specific_output_folder_path  # This is now like "outputs/run_nx16"
        )

        # Construct the full output filename that main.py should use
        # This ensures the pickled config has the exact filename.
        config.outputs.output_filename = os.path.join(
            config.outputs.output_path,  # "."
            config.outputs.output_folder,  # "outputs/run_nx16"
            f"{config.outputs.output_base_name}{config.outputs.output_suffix}{config.outputs.output_extension}",
        )
        # Example: ./outputs/run_nx16/_traveling_vortex_16x16.nc

        temp_config_filename = ""
        filepath = None
        try:
            # Pickle the modified config object to a temporary file
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".pkl", mode="wb"
            ) as tmp_file:
                pickle.dump(config, tmp_file)
                temp_config_filename = tmp_file.name

            # Pass the path of the pickled file to the run function
            # `config` here is passed for run_single_atmpy_case to log info from it.
            # The actual config used by main.py comes from temp_config_filename.
            filepath = run_single_atmpy_case(
                config, TEST_CASE_NAME, MAIN_SCRIPT_PATH, temp_config_filename
            )

        finally:
            # Clean up the temporary pickle file
            if temp_config_filename and os.path.exists(temp_config_filename):
                os.remove(temp_config_filename)

        if filepath:
            output_files[nx_current] = filepath
        else:
            logging.error(f"Aborting study due to failure in run Nx={nx_current}.")
            # Consider cleaning up BASE_OUTPUT_DIR or partially generated files if study aborts
            return
        time.sleep(1)

    logging.info("\n--- All Convergence Runs Completed ---")
    logging.info("Output files:")
    for nx_val, fpath in output_files.items():
        logging.info(f"Nx = {nx_val}: {fpath}")

    manifest_path = os.path.join(BASE_OUTPUT_DIR, "output_files_manifest.txt")
    with open(manifest_path, "w") as f:
        for nx_val, fpath in output_files.items():
            f.write(f"{nx_val}:{fpath}\n")
    logging.info(f"Manifest saved to {manifest_path}")
    logging.info(
        f"Now run the post_process_convergence.py script."
    )  # Removed reference to BASE_OUTPUT_DIR


if __name__ == "__main__":
    main()
