from __future__ import annotations

import time
import logging
import os
import glob
from typing import TYPE_CHECKING, Dict, Any, Optional, Tuple

import netCDF4  # Required for Solver.load_checkpoint_data
import numpy as np

# Assuming output_writer.py is in atmpy.io
from atmpy.io.output_writers import NetCDFWriter
from atmpy.infrastructure.enums import VariableIndices as VI

# Basic logging setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

if TYPE_CHECKING:
    from atmpy.configuration.simulation_configuration import SimulationConfig
    from atmpy.grid.kgrid import Grid
    from atmpy.variables.variables import Variables
    from atmpy.variables.multiple_pressure_variables import MPV
    from atmpy.time_integrators.abstract_time_integrator import AbstractTimeIntegrator


class Solver:
    """
    Orchestrates the atmospheric simulation loop, including output and checkpointing.
    """

    def __init__(
        self,
        config: "SimulationConfig",
        grid: "Grid",
        variables: "Variables",
        mpv: "MPV",
        time_integrator: "AbstractTimeIntegrator",
        initial_t: float = 0.0,
        initial_step: int = 0,
        # Add other components if needed
    ):
        self.config = config
        self.grid = grid
        self.variables = variables
        self.mpv = mpv
        self.time_integrator = time_integrator

        self.current_t: float = initial_t
        self.current_step: int = initial_step
        self.start_time_wall: float = 0.0  # Wall time for this run instance

        # Extract core temporal parameters
        self.dt: float = config.temporal.dtfixed
        if self.dt <= 0:
            raise ValueError("Solver requires positive dt.")
        self.tmax: float = config.temporal.tmax
        self.stepmax: int = config.temporal.stepmax
        self.tout: np.ndarray = np.array(config.temporal.tout)
        self._output_time_tolerance = (
            self.dt * 0.51
        )  # Tolerance for hitting output times

        # Analysis Output config
        self.output_frequency_steps: int = config.outputs.output_frequency_steps
        self.output_filename = self.config.outputs.output_filename
        self._output_times_saved = np.zeros_like(self.tout, dtype=bool)

        # Checkpoint config (assuming these exist in config.outputs)
        self.enable_checkpointing = getattr(
            config.outputs, "enable_checkpointing", True
        )  # Default to True
        self.checkpoint_frequency_steps = getattr(
            config.outputs, "checkpoint_frequency_steps", 100
        )
        self.checkpoint_filename = getattr(
            config.outputs, "checkpoint_filename"
        )  # Changed template
        self.checkpoint_keep_n = getattr(config.outputs, "checkpoint_keep_n", 2)

        logging.info(
            f"Solver initialized. Starting at t={self.current_t:.4f}, step={self.current_step}"
        )
        logging.info(
            f"  Output file: {self.output_filename} (freq: {self.output_frequency_steps} steps)"
        )
        logging.info(
            f"  Checkpointing: {self.enable_checkpointing} (freq: {self.checkpoint_frequency_steps} steps, base: {self.checkpoint_filename}, keep: {self.checkpoint_keep_n})"
        )

    def run(self) -> None:
        """Executes the main simulation loop."""
        logging.info(
            f"--- Starting simulation run from t={self.current_t:.4f}, step={self.current_step} ---"
        )
        self.start_time_wall = time.time()

        # --- Initial Output/Checkpoint (at initial state) ---
        self._handle_saving()

        # --- Main Time Loop ---
        while True:
            if self.current_t >= self.tmax - 1e-9:
                logging.info(f"Reached tmax = {self.tmax:.4f}. Stopping.")
                break
            if self.current_step >= self.stepmax:
                logging.info(f"Reached stepmax = {self.stepmax}. Stopping.")
                break

            step_start_time = time.time()
            try:
                self.time_integrator.step()
            except Exception as e:
                logging.error(
                    f"Error during step {self.current_step + 1} (t={self.current_t:.4f}): {e}",
                    exc_info=True,
                )
                self._save_checkpoint(force_save=True)
                raise e

            self.current_step += 1
            self.current_t += self.dt

            log_interval_steps = max(1, self.stepmax // 20) if self.stepmax > 0 else 50
            if self.current_step % log_interval_steps == 0:
                logging.info(
                    f"Step: {self.current_step}/{self.stepmax}, Time: {self.current_t:.4f}/{self.tmax:.4f}, Step Time: {time.time() - step_start_time:.4f}s"
                )

            self._handle_saving()

        # --- Finalization ---
        end_time_wall = time.time()
        total_time = end_time_wall - self.start_time_wall
        logging.info(f"--- Simulation finished ---")
        logging.info(
            f"Final state: Step={self.current_step}, Time={self.current_t:.4f}"
        )
        logging.info(f"Total wall clock time for this run: {total_time:.2f} seconds")
        # Consider saving final state explicitly if needed
        # self._save_output("final state")

    def _handle_saving(self) -> None:
        """Checks if output or checkpoint should be saved."""
        self._check_and_save_output()
        self._check_and_save_checkpoint()

    def _check_and_save_output(self) -> None:
        """Handles saving of standard analysis output."""
        save_now = False
        reason = ""
        current_step_is_output = (
            self.output_frequency_steps > 0
            and self.current_step % self.output_frequency_steps == 0
        )

        # Check specific output times
        for i, t_out in enumerate(self.tout):
            if (
                not self._output_times_saved[i]
                and abs(self.current_t - t_out) <= self._output_time_tolerance
            ):
                # Prevent saving again if restarting exactly at output time, unless it's also a freq step
                if (
                    self.current_t > 0
                    or self.current_step == 0
                    or current_step_is_output
                ):
                    save_now = True
                    self._output_times_saved[i] = True
                    reason = f"specific time t={t_out:.4f}"
                    break  # Prioritize saving for specific time

        # Check step-based frequency if not already saving for specific time
        if not save_now and current_step_is_output:
            save_now = True
            reason = f"step frequency ({self.output_frequency_steps})"

        # Check if it's the very first step (T=0)
        if not save_now and self.current_step == 0:
            save_now = True
            reason = "initial state (t=0)"

        if save_now:
            self._save_output(reason)

    def _save_output(self, reason: str) -> None:
        """Saves standard analysis output using NetCDFWriter."""
        logging.info(
            f"Saving analysis output to {self.output_filename} (Reason: {reason}) at t={self.current_t:.4f}, step={self.current_step}"
        )

        file_exists = os.path.exists(self.output_filename)
        mode = "a" if file_exists and self.current_step > 0 else "w"

        try:
            with NetCDFWriter(
                self.output_filename,
                self.grid,
                self.config,
                mode=mode,
                is_checkpoint=False,
            ) as writer:
                # --- Define Dimensions (based on INNER grid size) ---
                inner_cell_dims = tuple(dim_map_cells[i] for i in range(self.grid.ndim))
                inner_node_dims = tuple(dim_map_nodes[i] for i in range(self.grid.ndim))
                # Prepend 'time'
                time_inner_cell_dims = ("time",) + inner_cell_dims
                time_inner_node_dims = ("time",) + inner_node_dims

                # --- Define Variables ---
                # Helper dicts for attributes
                attrs = {
                    "rho": {"units": "kg m-3", "long_name": "Density"},
                    "rhoY": {
                        "units": "kg m-3 K",
                        "long_name": "rho * Potential Temperature",
                    },  # Adjust as needed
                    "u": {"units": "m s-1", "long_name": "X-velocity component"},
                    "v": {"units": "m s-1", "long_name": "Y-velocity component"},
                    "w": {"units": "m s-1", "long_name": "Z-velocity component"},
                    "p2_nodes": {
                        "units": "Pa K m3 kg-1",
                        "long_name": "Exner Pressure Perturbation (Nodes)",
                    },  # Adjust
                }
                vars_to_define = {
                    "rho": time_inner_cell_dims,
                    "rhoY": time_inner_cell_dims,
                    "u": time_inner_cell_dims,
                    "p2_nodes": time_inner_node_dims,
                }
                if self.grid.ndim >= 2:
                    vars_to_define["v"] = time_inner_cell_dims
                if self.grid.ndim >= 3:
                    vars_to_define["w"] = time_inner_cell_dims

                for name, dims in vars_to_define.items():
                    writer.define_variable(
                        name, dims, dtype=np.float64, attributes=attrs.get(name)
                    )

                # --- Prepare Data Dictionary (INNER domain data) ---
                inner_slice = self.grid.get_inner_slice()

                rho_inner = self.variables.cell_vars[inner_slice + (VI.RHO,)]
                safe_rho_inner = np.maximum(rho_inner, 1e-15)

                data_to_write = {
                    "rho": rho_inner,
                    "rhoY": self.variables.cell_vars[inner_slice + (VI.RHOY,)],
                    "u": self.variables.cell_vars[inner_slice + (VI.RHOU,)]
                    / safe_rho_inner,
                    "p2_nodes": self.mpv.p2_nodes[
                        inner_slice
                    ],  # Slice works for nodes too
                }
                if self.grid.ndim >= 2 and VI.RHOV < self.variables.num_vars_cell:
                    data_to_write["v"] = (
                        self.variables.cell_vars[inner_slice + (VI.RHOV,)]
                        / safe_rho_inner
                    )
                if self.grid.ndim >= 3 and VI.RHOW < self.variables.num_vars_cell:
                    data_to_write["w"] = (
                        self.variables.cell_vars[inner_slice + (VI.RHOW,)]
                        / safe_rho_inner
                    )

                # --- Write Data ---
                writer.write_timestep_data(self.current_t, data_to_write)

            logging.debug(f"Successfully wrote analysis output step.")

        except Exception as e:
            logging.error(
                f"Failed to save analysis output to {self.output_filename}: {e}",
                exc_info=True,
            )

    def _check_and_save_checkpoint(self) -> None:
        """Handles saving of checkpoint files."""
        if not self.enable_checkpointing:
            return
        if self.current_step % self.checkpoint_frequency_steps == 0:
            self._save_checkpoint()

    def _get_checkpoint_filename(self) -> str:
        """Generates the filename for the current checkpoint."""

        # Use simple formatting, assuming base_name is like '/path/to/chkpt_'
        return f"{self.checkpoint_filename}_step_{self.current_step:09d}.nc"

    def _manage_old_checkpoints(self):
        """Deletes older checkpoint files."""
        if self.checkpoint_keep_n <= 0:
            return
        try:
            pattern = f"{self.checkpoint_base_name}step*.nc"
            checkpoint_files = sorted(glob.glob(pattern))
            files_to_delete = checkpoint_files[: -self.checkpoint_keep_n]
            for f in files_to_delete:
                try:
                    os.remove(f)
                    logging.info(f"Removed old checkpoint file: {f}")
                except OSError as e:
                    logging.warning(f"Could not remove old checkpoint file {f}: {e}")
        except Exception as e:
            logging.error(f"Error during old checkpoint management: {e}", exc_info=True)

    def _save_checkpoint(self, force_save=False):
        """Saves the complete simulation state for restarting."""
        if not self.enable_checkpointing and not force_save:
            return

        chk_filename = self._get_checkpoint_filename()
        reason = "error recovery" if force_save else "scheduled"
        logging.info(
            f"Saving checkpoint to {chk_filename} (Reason: {reason}) at t={self.current_t:.4f}, step={self.current_step}"
        )

        try:
            with NetCDFWriter(
                chk_filename, self.grid, self.config, mode="w", is_checkpoint=True
            ) as writer:
                # --- Define Dimensions (based on FULL grid size) ---
                # These will be created automatically by the writer based on is_checkpoint=True
                full_cell_dims = tuple(dim_map_cells[i] for i in range(self.grid.ndim))
                full_node_dims = tuple(dim_map_nodes[i] for i in range(self.grid.ndim))

                # --- Define Variables ---
                writer.define_variable(
                    "cell_vars",
                    full_cell_dims + ("var_cell",),  # Add var dimension
                    dtype=self.variables.cell_vars.dtype,
                    attributes={"long_name": "Full cell-centered state variables"},
                )
                writer.define_variable(
                    "p2_nodes",
                    full_node_dims,
                    dtype=self.mpv.p2_nodes.dtype,
                    attributes={"long_name": "Full p2_nodes state"},
                )
                # Add other MPV state if needed
                # writer.define_variable('p2_cells', full_cell_dims, ...)
                # writer.define_variable('dp2_nodes', full_node_dims, ...)

                # --- Prepare Data (FULL arrays) ---
                data_to_write = {
                    "cell_vars": self.variables.cell_vars,  # Save the full array
                    "p2_nodes": self.mpv.p2_nodes,  # Save the full array
                    # Add others...
                }

                # --- Global Attributes for Checkpoint Info ---
                global_attrs = {
                    "checkpoint_time": self.current_t,
                    "checkpoint_step": self.current_step,
                    "description": f"Atmpy Checkpoint for step {self.current_step}",
                    # Optional: Save config hash or critical params?
                }

                # --- Write Snapshot ---
                writer.write_snapshot_data(data_to_write, global_attrs)

            logging.debug(f"Successfully wrote checkpoint file: {chk_filename}")
            if not force_save:
                self._manage_old_checkpoints()

        except Exception as e:
            logging.error(
                f"Failed to save checkpoint to {chk_filename}: {e}", exc_info=True
            )

    @staticmethod
    def load_checkpoint_data(filename: str) -> Dict[str, Any]:
        """Loads simulation state data from a checkpoint file."""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Checkpoint file not found: {filename}")

        logging.info(f"Loading state from checkpoint: {filename}")
        loaded_state = {}
        try:
            with netCDF4.Dataset(filename, "r") as ds:
                if (
                    "checkpoint_time" not in ds.ncattrs()
                    or "checkpoint_step" not in ds.ncattrs()
                ):
                    raise KeyError("Checkpoint missing time/step attributes.")
                loaded_state["current_t"] = ds.getncattr("checkpoint_time")
                loaded_state["current_step"] = ds.getncattr("checkpoint_step")

                required_vars = ["cell_vars", "p2_nodes"]  # Add others if saved
                for var_name in required_vars:
                    if var_name not in ds.variables:
                        raise KeyError(
                            f"Checkpoint missing required variable: '{var_name}'"
                        )
                    loaded_state[var_name] = ds.variables[var_name][:]

                logging.info(
                    f"Successfully loaded state for step {loaded_state['current_step']}, t={loaded_state['current_t']:.4f}"
                )
        except Exception as e:
            logging.error(
                f"Failed to load checkpoint file {filename}: {e}", exc_info=True
            )
            raise IOError(f"Error reading checkpoint file {filename}: {e}") from e
        return loaded_state


# --- Helper dictionaries used above ---
dim_map_cells = {0: "x", 1: "y", 2: "z"}
dim_map_nodes = {0: "x_node", 1: "y_node", 2: "z_node"}
if __name__ == "__main__":
    import argparse
    from atmpy.test_cases.traveling_vortex import TravelingVortex
    from atmpy.physics.thermodynamics import Thermodynamics
    from atmpy.grid.utility import DimensionSpec, create_grid
    from atmpy.variables.variables import Variables
    from atmpy.infrastructure.enums import (
        VariableIndices as VI,
        HydrostateIndices as HI,
        BoundaryConditions as BdryType,
        BoundarySide,
        PressureSolvers,
        DiscreteOperators,
        LinearSolvers,
        AdvectionRoutines,
        TimeIntegrators,
        Preconditioners,
    )
    from atmpy.boundary_conditions.bc_extra_operations import (
        WallAdjustment,
        PeriodicAdjustment,
    )
    from atmpy.variables.multiple_pressure_variables import MPV
    from atmpy.boundary_conditions.boundary_manager import BoundaryManager
    from atmpy.physics.eos import ExnerBasedEOS
    from atmpy.flux.flux import Flux
    from atmpy.pressure_solver.contexts import (
        DiscreteOperatorsContext,
        PressureContext,
    )
    from atmpy.time_integrators.contexts import TimeIntegratorContext
    from atmpy.pressure_solver.classical_pressure_solvers import ClassicalPressureSolver

    np.set_printoptions(linewidth=300)
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=10)

    ################################### Parser #########################################################################
    parser = argparse.ArgumentParser(description="Atmpy Simulation")
    parser.add_argument(
        "--restart", type=str, help="Path to checkpoint file to restart from."
    )
    args = parser.parse_args()

    #################################### Instantiate Test Case and Get Config ##########################################
    case = TravelingVortex()
    config = case.config  # The SimulationConfig object is now held by the case

    # Modify config if needed (e.g., simulation time)
    config.temporal.tmax = 0.05  # Run for a short time for testing
    config.temporal.stepmax = 101  # Limit steps
    config.outputs.output_frequency_steps = 2  # Output every 2 steps
    config.temporal.tout = [0.025]  # Also output at a specific time

    ##################################### Create Simulation Components based on Config #################################
    grid = config.grid
    gravity_vec = config.physics.gravity_strength
    Msq = config.model_regimes.Msq
    th = Thermodynamics()
    th.update(config.global_constants.gamma)

    ##################################### MPV ##########################################################################
    mpv = MPV(grid)

    ##################################### Variables ####################################################################
    num_vars = 6
    variables = Variables(
        grid, num_vars_cell=num_vars, num_vars_node=1
    )  # Adjust num_vars if needed

    # --- Initialize Solution using Test Case
    case.initialize_solution(variables, mpv)  # This now uses the case object

    ##################################### Boundary Manager #############################################################
    bm_config = config.get_boundary_manager_config()
    manager = BoundaryManager(bm_config)
    # Apply initial BCs (though initialize_solution might handle inner domain, ghosts need update)
    manager.apply_boundary_on_all_sides(variables.cell_vars)
    # nodal_bc_contexts = ([manager.get_context_for_side(i, is_nodal=True) for i in range(grid.ndim * 2)]) # Get contexts for p2
    # manager.apply_boundary_on_single_var_all_sides(mpv.p2_nodes, nodal_bc_contexts)

    ###################################### Flux ########################################################################
    eos = ExnerBasedEOS()
    flux = Flux(grid, variables, eos)

    ###################################### Pressure Solver #############################################################
    op_context = DiscreteOperatorsContext(
        operator_type=DiscreteOperators.CLASSIC_OPERATOR, grid=grid
    )
    linear_solver = LinearSolvers.BICGSTAB

    ps_context: PressureContext[ClassicalPressureSolver] = PressureContext(
        solver_type=PressureSolvers.CLASSIC_PRESSURE_SOLVER,
        op_context=op_context,
        linear_solver_type=linear_solver,
        precondition_type=Preconditioners.DIAGONAL,
        extra_dependencies={
            "grid": grid,
            "variables": variables,
            "mpv": mpv,
            "boundary_manager": manager,
            "coriolis": config.physics.coriolis,  # Get from config
            "Msq": Msq,
            "thermodynamics": th,
        },
    )
    pressure_solver = ps_context.instantiate()

    ######################################### Time Integrator ##########################################################
    from atmpy.time_integrators.imex_operator_splitting import IMEXTimeIntegrator

    ti_context: TimeIntegratorContext[IMEXTimeIntegrator] = TimeIntegratorContext(
        integrator_type=TimeIntegrators.IMEX,
        grid=grid,
        variables=variables,
        flux=flux,
        boundary_manager=manager,
        advection_routine=config.numerics.advection_routine,  # Get from config
        dt=config.temporal.dtfixed,  # Get from config
        extra_dependencies={
            "mpv": mpv,
            "pressure_solver": pressure_solver,
            "wind_speed": config.physics.wind_speed,  # Get from config
            "is_nonhydrostatic": config.model_regimes.is_nonhydrostatic,
            "is_nongeostrophic": config.model_regimes.is_nongeostrophic,
            "is_compressible": config.model_regimes.is_compressible,
        },
    )
    time_integrator = ti_context.instantiate()

    #     #################################### Optional(Initial Projection) ##################################################
    #     # 5. Perform Initial Projection (Optional)
    #     if config.numerics.initial_projection:
    #         print("Performing initial projection...")
    #         # This step needs careful implementation. It's essentially applying the
    #         # pressure correction mechanism once at t=0.
    #         # Option A: Add a dedicated method to the PressureSolver
    #         # pressure_solver.perform_initial_projection(variables, mpv, config.temporal.dtfixed0) # Pass initial dt if needed
    #         # Option B: Add a dedicated method to the TimeIntegrator
    #         # time_integrator.perform_initial_projection()
    #         # Option C: Implement the logic directly here (less ideal, duplicates solver logic)
    #
    #         # Let's assume Option A exists:
    #         try:
    #             # We need a way to call the implicit pressure update logic *once*
    #             # In Atmpy's IMEX integrator, this logic is inside `backward_update_implicit`.
    #             # We might need to refactor slightly or add a dedicated method.
    #             # A potential approach:
    #             logging.info(
    #                 "Applying initial projection via backward_update_implicit (check if dt=0 or small dt is appropriate)")
    #             # Temporarily modify state if needed (like PyBella subtracting background wind)
    #             # variables.adjust_background_wind(config.physics.wind_speed, scale=-1.0, in_place=True) # Example if needed
    #
    #             # Call the core implicit update. The dt might need consideration (e.g., dt=0 or a small initial dt?)
    #             # Using the initial dt (dtfixed0) seems plausible based on PyBella.
    #             time_integrator.backward_update_implicit(config.temporal.dtfixed0)  # Pass initial dt
    #
    #             # Restore state if modified
    #             # variables.adjust_background_wind(config.physics.wind_speed, scale=1.0, in_place=True) # Example if needed
    #
    #             # Re-apply BCs after projection modifies the state
    #             print("Re-applying boundary conditions after initial projection...")
    #             manager.apply_boundary_on_all_sides(variables.cell_vars)
    #             manager.apply_pressure_boundary_on_all_sides(mpv.p2_nodes, nodal_bc_contexts)
    #
    #         except AttributeError:
    #             print(
    #                 "Warning: Initial projection requested but corresponding method not found/implemented in pressure solver/time integrator.")
    #         except Exception as e:
    #             print(f"Error during initial projection: {e}")

    ################################## Restarting From Checkpoint ######################################################
    initial_t = 0.0
    initial_step = 0

    if args.restart:
        try:
            # --- Load state from checkpoint ---
            loaded_data = Solver.load_checkpoint_data(args.restart)
            initial_t = loaded_data["current_t"]
            # IMPORTANT: Increment step number to start *after* the loaded step
            initial_step = loaded_data["current_step"] + 1  # Start next step

            # --- Populate variables with loaded data ---
            if variables.cell_vars.shape == loaded_data["cell_vars"].shape:
                variables.cell_vars[...] = loaded_data["cell_vars"]
            else:
                raise ValueError("Checkpoint cell_vars shape mismatch!")
            if mpv.p2_nodes.shape == loaded_data["p2_nodes"].shape:
                mpv.p2_nodes[...] = loaded_data["p2_nodes"]
            else:
                raise ValueError("Checkpoint p2_nodes shape mismatch!")
            # ... load other state like mpv.p2_cells if saved ...

            logging.info(
                f"Restarting simulation from checkpoint state at t={initial_t:.4f}, step={initial_step}"
            )
            # No need to call case.initialize_solution or apply initial BCs/projection

        except (FileNotFoundError, IOError, KeyError, ValueError) as e:
            logging.error(f"Failed to restart from {args.restart}: {e}")
            logging.error("Exiting.")
            exit(1)
    else:
        # --- Standard Initialization ---
        logging.info("Starting new simulation.")
        case.initialize_solution(variables, mpv)
        manager.apply_boundary_on_all_sides(variables.cell_vars)
        # ... apply MPV BCs ...
        if config.numerics.initial_projection:
            # ... perform initial projection ...
            # ... reapply BCs ...
            pass

    ##################################### Create and Run the Solver ####################################################
    solver = Solver(
        config=config,
        grid=grid,
        variables=variables,
        mpv=mpv,
        time_integrator=time_integrator,
        initial_t=initial_t,  # Pass loaded/initial time
        initial_step=initial_step,  # Pass loaded/initial step
    )
    solver.run()

    # --- Post-processing / Analysis (Optional) ---
    print("\n --- Final State (Inner Domain Rho Example) ---")
    inner_slice = grid.get_inner_slice()
    print(variables.cell_vars[inner_slice + (VI.RHO,)])

    print("\n --- Final State (Inner Domain p2 Example) ---")
    print(mpv.p2_nodes[inner_slice])
