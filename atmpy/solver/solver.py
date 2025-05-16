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
from atmpy.infrastructure.enums import (
    VariableIndices as VI,
    PrimitiveVariableIndices as PVI,
)
from atmpy.physics.eos import ExnerBasedEOS
from atmpy.solver.utility import calculate_dynamic_dt

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

# Helper Dictionaries
dim_map_cells = {0: "x", 1: "y", 2: "z"}
dim_map_nodes = {0: "x_node", 1: "y_node", 2: "z_node"}

MACHINE_EPSILON = np.finfo(float).eps


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
            logging.info(
                f" Current time t={self.current_t:.4f}, step={self.current_step}"
            )
            if self.current_t >= self.tmax - 1e-9:
                logging.info(f"Reached tmax = {self.tmax:.4f}. Stopping.")
                break
            if self.current_step >= self.stepmax:
                logging.info(f"Reached stepmax = {self.stepmax}. Stopping.")
                break

            # --- Determine next output time for dt adjustment ---
            # Find the smallest t_out in self.tout that is > self.current_t
            # and hasn't been saved yet.
            # If all specific tout are done, use tmax as the next target.
            pending_output_times = self.tout[np.logical_not(self._output_times_saved)]
            future_output_times = pending_output_times[
                pending_output_times > self.current_t + MACHINE_EPSILON
            ]

            if future_output_times.size > 0:
                next_target_output_t = np.min(future_output_times)
            else:
                next_target_output_t = self.tmax

            # --- Calculate dynamic dt ---
            current_dt = calculate_dynamic_dt(
                self.variables,
                self.grid,
                self.config,
                self.current_t,
                next_target_output_t,
                self.current_step,
            )
            # Update the time integrator's dt if it stores it internally
            self.time_integrator.dt = current_dt

            step_start_time = time.time()
            try:
                self.time_integrator.step(global_step=self.current_step)
            except Exception as e:
                logging.error(
                    f"Error during step {self.current_step + 1} (t={self.current_t:.4f}): {e}",
                    exc_info=True,
                )
                self._save_checkpoint(force_save=True)
                raise e

            self.current_step += 1
            self.current_t += current_dt

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
        """Saves standard analysis output using NetCDFWriter, including conservative and primitive variables."""
        logging.info(
            f"Saving analysis output to {self.output_filename} (Reason: {reason}) at t={self.current_t:.4f}, step={self.current_step}"
        )

        file_exists = os.path.exists(self.output_filename)
        # Append if file exists and not the first step (t=0, step=0)
        mode = (
            "a"
            if file_exists and (self.current_t > 1e-9 or self.current_step > 0)
            else "w"
        )

        try:
            with NetCDFWriter(
                self.output_filename,
                self.grid,  # Pass the kgrid.Grid object
                self.config,
                mode=mode,
                is_checkpoint=False,
            ) as writer:
                # --- Define Dimensions (based on INNER grid size) ---
                # These are determined by the writer based on is_checkpoint=False
                inner_cell_dims = []
                for i in range(self.grid.ndim):
                    inner_cell_dims.append(
                        dim_map_cells[i]
                    )  # Using global DIM_MAP_CELLS from output_writers
                inner_cell_dims = tuple(inner_cell_dims)

                # Prepend 'time'
                time_inner_cell_dims = ("time",) + inner_cell_dims
                # Node dimensions would be similar if you save nodal primitive vars
                # inner_node_dims = tuple(DIM_MAP_NODES[i] for i in range(self.grid.ndim))
                # time_inner_node_dims = ("time",) + inner_node_dims

                # --- Define Variables ---
                # Helper dicts for attributes
                attrs_conservative = {
                    "rho": {"units": "kg m-3", "long_name": "Density (conservative)"},
                    "rhou": {
                        "units": "kg m-2 s-1",
                        "long_name": "X-momentum density (rho*u)",
                    },
                    "rhov": {
                        "units": "kg m-2 s-1",
                        "long_name": "Y-momentum density (rho*v)",
                    },
                    "rhow": {
                        "units": "kg m-2 s-1",
                        "long_name": "Z-momentum density (rho*w)",
                    },
                    "rhoY": {
                        "units": "kg m-3 K",
                        "long_name": "Potential temperature density (rho*Theta_nd or rho*Y)",
                    },
                    "rhoX": {"units": "kg m-3", "long_name": "Tracer density (rho*X)"},
                }
                attrs_primitive = {
                    "p": {
                        "units": (
                            "Pa K^gamma"
                            if isinstance(self.time_integrator.flux.eos, ExnerBasedEOS)
                            else "Pa"
                        ),
                        "long_name": "Pressure (Exner or Thermodynamic)",
                    },
                    "u": {
                        "units": "m s-1",
                        "long_name": "X-velocity component (primitive)",
                    },
                    "v": {
                        "units": "m s-1",
                        "long_name": "Y-velocity component (primitive)",
                    },
                    "w": {
                        "units": "m s-1",
                        "long_name": "Z-velocity component (primitive)",
                    },
                    "Y": {
                        "units": "K",
                        "long_name": "Potential temperature (Theta_nd or Y)",
                    },
                    # Or dimensionless if Y is used directly
                    "X": {"units": "kg/kg", "long_name": "Tracer mixing ratio (X)"},
                    # You might also want to save the actual temperature T
                    "T": {"units": "K", "long_name": "Temperature"},
                }
                # Add p2_nodes if you still want to save it
                attrs_nodal = {
                    "p2_nodes": {
                        "units": "Pa K m3 kg-1",  # Adjust if definition changed
                        "long_name": "Exner Pressure Perturbation (Nodes)",
                    },
                }

                vars_to_define = {}
                # Conservative
                vars_to_define["rho"] = (
                    time_inner_cell_dims,
                    attrs_conservative["rho"],
                )
                vars_to_define["rhou"] = (
                    time_inner_cell_dims,
                    attrs_conservative["rhou"],
                )
                if self.grid.ndim >= 2 and VI.RHOV < self.variables.num_vars_cell:
                    vars_to_define["rhov"] = (
                        time_inner_cell_dims,
                        attrs_conservative["rhov"],
                    )
                if self.grid.ndim >= 3 and VI.RHOW < self.variables.num_vars_cell:
                    vars_to_define["rhow"] = (
                        time_inner_cell_dims,
                        attrs_conservative["rhow"],
                    )
                vars_to_define["rhoY"] = (
                    time_inner_cell_dims,
                    attrs_conservative["rhoY"],
                )
                if VI.RHOX < self.variables.num_vars_cell:  # Check if RHOX is used
                    vars_to_define["rhoX"] = (
                        time_inner_cell_dims,
                        attrs_conservative["rhoX"],
                    )

                # Primitive (ensure PVI enum is defined and used correctly)
                vars_to_define["p"] = (time_inner_cell_dims, attrs_primitive["p"])
                vars_to_define["u"] = (time_inner_cell_dims, attrs_primitive["u"])
                if (
                    self.grid.ndim >= 2 and PVI.V < self.variables.num_vars_cell
                ):  # Assuming num_vars_cell ~ num_primitive_vars
                    vars_to_define["v"] = (time_inner_cell_dims, attrs_primitive["v"])
                if self.grid.ndim >= 3 and PVI.W < self.variables.num_vars_cell:
                    vars_to_define["w"] = (time_inner_cell_dims, attrs_primitive["w"])
                vars_to_define["Y"] = (
                    time_inner_cell_dims,
                    attrs_primitive["Y"],
                )  # Renamed to Y_prim to avoid clash with y-coord
                if (
                    PVI.X < self.variables.num_vars_cell
                ):  # Check if X is used in primitives
                    vars_to_define["X"] = (
                        time_inner_cell_dims,
                        attrs_primitive["X"],
                    )  # Renamed to X_prim
                vars_to_define["T"] = (time_inner_cell_dims, attrs_primitive["T"])

                # Nodal (if any)
                # Example: p2_nodes
                # if self.mpv and hasattr(self.mpv, 'p2_nodes'):
                inner_node_dims = tuple(dim_map_nodes[i] for i in range(self.grid.ndim))
                time_inner_node_dims = ("time",) + inner_node_dims
                vars_to_define["p2_nodes"] = (
                    time_inner_node_dims,
                    attrs_nodal["p2_nodes"],
                )

                for name, (dims, attributes) in vars_to_define.items():
                    # Ensure all dimensions in 'dims' exist before defining
                    all_dims_exist = True
                    for d_name in dims:
                        if d_name not in writer.ds.dimensions:
                            logging.warning(
                                f"Dimension '{d_name}' for variable '{name}' not found. Skipping variable definition."
                            )
                            all_dims_exist = False
                            break
                    if all_dims_exist:
                        writer.define_variable(
                            name, dims, dtype=np.float64, attributes=attributes
                        )
                    else:
                        logging.warning(
                            f"Variable '{name}' was not defined due to missing dimensions."
                        )

                # --- Prepare Data Dictionary (INNER domain data) ---
                inner_slice = self.grid.get_inner_slice()
                eos = (
                    self.time_integrator.flux.eos
                )  # Get EOS from time integrator or config

                # Calculate primitives first
                # self.variables.to_primitive(
                #     eos
                # )  # This populates self.variables.primitives

                data_to_write = {}

                # Conservative Variables
                data_to_write["rho"] = self.variables.cell_vars[inner_slice + (VI.RHO,)]
                data_to_write["rhou"] = self.variables.cell_vars[
                    inner_slice + (VI.RHOU,)
                ]
                if self.grid.ndim >= 2 and VI.RHOV < self.variables.num_vars_cell:
                    data_to_write["rhov"] = self.variables.cell_vars[
                        inner_slice + (VI.RHOV,)
                    ]
                if self.grid.ndim >= 3 and VI.RHOW < self.variables.num_vars_cell:
                    data_to_write["rhow"] = self.variables.cell_vars[
                        inner_slice + (VI.RHOW,)
                    ]
                data_to_write["rhoY"] = self.variables.cell_vars[
                    inner_slice + (VI.RHOY,)
                ]
                if VI.RHOX < self.variables.num_vars_cell:
                    data_to_write["rhoX"] = self.variables.cell_vars[
                        inner_slice + (VI.RHOX,)
                    ]

                # Primitive Variables
                # Ensure PVI enum indices are correct for your self.variables.primitives array
                rho = self.variables.cell_vars[inner_slice + (VI.RHO,)]
                # data_to_write["p"] = self.variables.cell_vars[inner_slice + (PVI.P,)]
                u = self.variables.cell_vars[inner_slice + (VI.RHOU,)] / rho
                from tempfile import TemporaryFile

                data_to_write["u"] = (
                    self.variables.cell_vars[inner_slice + (VI.RHOU,)] / rho
                )
                if self.grid.ndim >= 2 and PVI.V < self.variables.num_vars_cell:
                    data_to_write["v"] = (
                        self.variables.cell_vars[inner_slice + (VI.RHOV,)] / rho
                    )
                if self.grid.ndim >= 3 and PVI.W < self.variables.num_vars_cell:
                    data_to_write["w"] = (
                        self.variables.cell_vars[inner_slice + (VI.RHOW,)] / rho
                    )
                data_to_write["Y"] = (
                    self.variables.cell_vars[inner_slice + (VI.RHOY,)] / rho
                )
                if PVI.X < self.variables.num_vars_cell:
                    data_to_write["X"] = (
                        self.variables.cell_vars[inner_slice + (VI.RHOX,)] / rho
                    )

                # Nodal variables (if any)
                # if "p2_nodes" in vars_to_define: # Check if it was successfully defined
                #    if self.mpv and hasattr(self.mpv, 'p2_nodes'):
                data_to_write["p2_nodes"] = self.mpv.p2_nodes[
                    inner_slice
                ]  # Slicing works for nodes too

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
            pattern = f"{self.checkpoint_filename}_step*.nc"
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
